//! This module contains all code relevant to Voice within Warp.
//!
//! Voice is used for voice input within Warp.

// Warp Wave 6-1: `pub(crate) mod transcribe` was physically deleted along with `ServerApi::transcribe`.
// Its submodules `transcribe/api/{request,response}` only held the wire types of the removed
// cloud `/ai/transcribe` endpoint. Local voice uses the `voice/transcriber.rs::Transcriber`
// trait + `TranscribeError`.
