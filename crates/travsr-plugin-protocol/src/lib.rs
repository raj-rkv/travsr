#![forbid(unsafe_code)]
//! travsr-plugin-protocol — Plugin trait, wire message types, and frame codec.

pub mod codec;
pub mod embed;
pub mod ffi_marker;
pub mod language_map;
pub mod plugin;
pub mod types;

pub use codec::{decode_message, encode_message, write_message};
pub use embed::{
    EmbedHandshakeRequest, EmbedHandshakeResponse, EmbedPlugin, EmbedPluginRequest,
    EmbedPluginResponse, EmbedRequest, EmbedResponse, KnnRequest, KnnResponse, Space,
    EMBED_PROTOCOL_VERSION,
};
pub use ffi_marker::{FfiMarker, FfiMarkerKind};
pub use language_map::{language_from_proto_str, language_to_proto_str};
pub use plugin::Plugin;
pub use travsr_core::UnresolvedCall;
pub use types::{
    DiagnosticSeverity, HandshakeRequest, HandshakeResponse, InvokeRequest, InvokeResponse,
    ParseRequest, ParseResponse, PluginDiagnostic, PluginError, PluginRequest, PluginResponse,
    PROTOCOL_VERSION,
};
