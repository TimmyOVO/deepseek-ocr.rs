use rocket::{
    Request, Response,
    fairing::{Fairing, Info, Kind},
    http::{Header, Method, Status},
};

pub struct Cors;

#[rocket::async_trait]
impl Fairing for Cors {
    fn info(&self) -> Info {
        Info {
            name: "CORS headers",
            kind: Kind::Response,
        }
    }

    async fn on_response<'r>(&self, req: &'r Request<'_>, res: &mut Response<'r>) {
        res.set_header(Header::new("Access-Control-Allow-Origin", "*"));
        res.set_header(Header::new(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        ));
        let allow_headers = req
            .headers()
            .get_one("Access-Control-Request-Headers")
            .unwrap_or("Authorization, Content-Type");
        res.set_header(Header::new("Access-Control-Allow-Headers", allow_headers));
        res.set_header(Header::new("Access-Control-Max-Age", "86400"));

        if req.method() == Method::Options && res.status() == Status::NotFound {
            res.set_status(Status::Ok);
        }
    }
}
