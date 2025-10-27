use anyhow::Error;
use rocket::{
    http::Status,
    response::{Responder, status::Custom},
    serde::json::Json,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Internal(String),
}

impl From<Error> for ApiError {
    fn from(err: Error) -> Self {
        ApiError::Internal(format!("{err:#}"))
    }
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    r#type: String,
}

impl<'r> Responder<'r, 'static> for ApiError {
    fn respond_to(self, request: &'r rocket::Request<'_>) -> rocket::response::Result<'static> {
        let (status, error_type) = match self {
            ApiError::BadRequest(_) => (Status::BadRequest, "invalid_request_error"),
            ApiError::Internal(_) => (Status::InternalServerError, "internal_error"),
        };
        let body = ErrorBody {
            error: ErrorDetail {
                message: self.to_string(),
                r#type: error_type.to_string(),
            },
        };
        Custom(status, Json(body)).respond_to(request)
    }
}
