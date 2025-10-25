use deepseek_ocr_core::conversation::get_conv_template;

#[test]
fn conversation_deepseek_prompt_contains_expected_markers() {
    let mut conv = get_conv_template("deepseek").expect("template registered");
    conv.append_message(conv.roles.0.clone(), Some("Hello!".to_string()));
    conv.append_message(conv.roles.1.clone(), Some("Hi! This is Tony.".to_string()));
    conv.append_message(conv.roles.0.clone(), Some("Who are you?".to_string()));
    conv.append_message(
        conv.roles.1.clone(),
        Some("I am a helpful assistant.".to_string()),
    );
    conv.append_message(conv.roles.0.clone(), Some("How are you?".to_string()));
    conv.append_message(conv.roles.1.clone(), None);
    let prompt = conv.get_prompt();
    assert!(prompt.contains("Hello!"));
    assert!(prompt.contains("<｜end▁of▁sentence｜>"));
}
