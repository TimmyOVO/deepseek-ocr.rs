use std::{
    collections::BTreeMap,
    sync::{RwLock, RwLockReadGuard},
};

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeparatorStyle {
    DeepSeek,
    DeepSeekV2,
    Plain,
    Alignment,
}

#[derive(Debug, Clone)]
pub struct ConversationTemplate {
    pub name: String,
    pub system_template: String,
    pub system_message: String,
    pub roles: (String, String),
    pub messages: Vec<(String, Option<String>)>,
    pub offset: usize,
    pub sep_style: SeparatorStyle,
    pub sep: String,
    pub sep2: Option<String>,
    pub stop_str: Vec<String>,
    pub stop_token_ids: Vec<i64>,
}

impl ConversationTemplate {
    pub fn set_system_message<S: Into<String>>(&mut self, system_message: S) {
        self.system_message = system_message.into();
    }

    pub fn append_message<S>(&mut self, role: S, message: Option<String>)
    where
        S: Into<String>,
    {
        self.messages.push((role.into(), message));
    }

    pub fn update_last_message<S: Into<String>>(&mut self, message: S) {
        if let Some((_, slot)) = self.messages.last_mut() {
            *slot = Some(message.into());
        }
    }

    pub fn reset_messages(&mut self) {
        self.messages.clear();
    }

    pub fn get_prompt(&self) -> String {
        match self.sep_style {
            SeparatorStyle::DeepSeek => self.render_deepseek(),
            SeparatorStyle::DeepSeekV2 => self.render_deepseek_v2(),
            SeparatorStyle::Plain => self.render_plain(),
            SeparatorStyle::Alignment => self.render_alignment(),
        }
    }

    fn render_deepseek(&self) -> String {
        let seps = [self.sep.as_str(), self.sep2.as_deref().unwrap_or_default()];
        let system_prompt = self
            .system_template
            .replace("{system_message}", &self.system_message);
        let mut buffer = String::new();
        if !system_prompt.is_empty() {
            buffer.push_str(&system_prompt);
            buffer.push_str(seps[0]);
        }
        for (idx, (role, message)) in self.messages.iter().enumerate() {
            match message.as_ref().map(|m| m.trim()).filter(|m| !m.is_empty()) {
                Some(content) => {
                    buffer.push_str(role);
                    buffer.push_str(": ");
                    buffer.push_str(content);
                    buffer.push_str(seps[idx % 2]);
                }
                None => {
                    buffer.push_str(role);
                    buffer.push(':');
                }
            }
        }
        buffer
    }

    fn render_deepseek_v2(&self) -> String {
        let seps = [self.sep.as_str(), self.sep2.as_deref().unwrap_or_default()];
        let system_prompt = self
            .system_template
            .replace("{system_message}", &self.system_message);
        let mut buffer = String::new();
        if !system_prompt.is_empty() {
            buffer.push_str(&system_prompt);
            buffer.push_str(seps[0]);
        }

        for (role, message) in &self.messages {
            if let Some(content) = message.as_ref().map(|m| m.trim()).filter(|m| !m.is_empty()) {
                if role == "User" {
                    buffer.push_str("<｜sft▁begin｜>\n");
                    buffer.push_str(content);
                    buffer.push_str(seps[0]);
                } else {
                    buffer.push_str(content);
                    buffer.push_str(seps[1]);
                }
            }
        }
        buffer
    }

    fn render_plain(&self) -> String {
        let seps = [self.sep.as_str(), self.sep2.as_deref().unwrap_or_default()];
        let mut buffer = String::new();
        for (idx, (_, message)) in self.messages.iter().enumerate() {
            if let Some(content) = message.as_ref().map(|m| m.trim()).filter(|m| !m.is_empty()) {
                buffer.push_str(content);
                buffer.push_str(seps[idx % 2]);
            }
        }
        buffer
    }

    fn render_alignment(&self) -> String {
        let seps = [self.sep.as_str(), self.sep2.as_deref().unwrap_or_default()];
        let mut buffer = String::new();
        for (idx, (_, message)) in self.messages.iter().enumerate() {
            match message.as_ref().map(|m| m.trim()).filter(|m| !m.is_empty()) {
                Some(content) => {
                    if idx % 2 == 0 {
                        buffer.push_str("<image>\n");
                        buffer.push_str(seps[idx % 2]);
                    } else {
                        buffer.push_str(content);
                        buffer.push_str(seps[idx % 2]);
                    }
                }
                None => {}
            }
        }
        buffer
    }
}

impl Default for ConversationTemplate {
    fn default() -> Self {
        ConversationTemplate {
            name: String::new(),
            system_template: "{system_message}".into(),
            system_message: String::new(),
            roles: ("USER".into(), "ASSISTANT".into()),
            messages: Vec::new(),
            offset: 0,
            sep_style: SeparatorStyle::DeepSeek,
            sep: "\n".into(),
            sep2: None,
            stop_str: Vec::new(),
            stop_token_ids: Vec::new(),
        }
    }
}

static CONVERSATION_TEMPLATES: Lazy<RwLock<BTreeMap<String, ConversationTemplate>>> =
    Lazy::new(|| {
        let mut map = BTreeMap::new();
        map.insert("deepseek".into(), deepseek_template());
        map.insert("deepseekv2".into(), deepseek_v2_template());
        map.insert("plain".into(), plain_template());
        map.insert("alignment".into(), alignment_template());
        RwLock::new(map)
    });

pub fn register_conv_template(template: ConversationTemplate, override_existing: bool) {
    let mut guard = CONVERSATION_TEMPLATES
        .write()
        .expect("conversation registry poisoned");
    if !override_existing && guard.contains_key(&template.name) {
        panic!("{} has been registered", template.name);
    }
    guard.insert(template.name.clone(), template);
}

pub fn get_conv_template(name: &str) -> Option<ConversationTemplate> {
    let guard: RwLockReadGuard<_> = CONVERSATION_TEMPLATES
        .read()
        .expect("conversation registry poisoned");
    guard.get(name).cloned()
}

fn deepseek_template() -> ConversationTemplate {
    ConversationTemplate {
        name: "deepseek".into(),
        system_template: "{system_message}".into(),
        system_message: String::new(),
        roles: ("<|User|>".into(), "<|Assistant|>".into()),
        messages: Vec::new(),
        offset: 0,
        sep_style: SeparatorStyle::DeepSeek,
        sep: "\n\n".into(),
        sep2: Some("<｜end▁of▁sentence｜>".into()),
        stop_str: vec!["User:".into(), "<｜end▁of▁sentence｜>".into()],
        stop_token_ids: vec![100001],
    }
}

fn deepseek_v2_template() -> ConversationTemplate {
    ConversationTemplate {
        name: "deepseekv2".into(),
        system_template: "{system_message}".into(),
        system_message: String::new(),
        roles: ("<｜User｜>".into(), "<｜Assistant｜>".into()),
        messages: Vec::new(),
        offset: 0,
        sep_style: SeparatorStyle::DeepSeek,
        sep: "".into(),
        sep2: Some("<｜end▁of▁sentence｜>".into()),
        stop_str: vec!["User:".into(), "<｜end▁of▁sentence｜>".into()],
        stop_token_ids: vec![100001],
    }
}

fn plain_template() -> ConversationTemplate {
    ConversationTemplate {
        name: "plain".into(),
        system_template: String::new(),
        system_message: String::new(),
        roles: ("".into(), "".into()),
        messages: Vec::new(),
        offset: 0,
        sep_style: SeparatorStyle::Plain,
        sep: "".into(),
        sep2: Some("".into()),
        stop_str: vec!["</s>".into()],
        stop_token_ids: vec![100001],
    }
}

fn alignment_template() -> ConversationTemplate {
    ConversationTemplate {
        name: "alignment".into(),
        system_template: String::new(),
        system_message: String::new(),
        roles: ("".into(), "".into()),
        messages: Vec::new(),
        offset: 0,
        sep_style: SeparatorStyle::Alignment,
        sep: "".into(),
        sep2: Some("".into()),
        stop_str: vec!["</s>".into()],
        stop_token_ids: vec![100001],
    }
}
