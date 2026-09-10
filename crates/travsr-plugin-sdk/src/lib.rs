#![forbid(unsafe_code)]
//! travsr-plugin-sdk — ergonomic plugin authoring for the Travsr plugin system.

pub use travsr_plugin_protocol::{
    DiagnosticSeverity, FfiMarker, FfiMarkerKind, HandshakeRequest, HandshakeResponse,
    InvokeRequest, InvokeResponse, ParseRequest, ParseResponse, Plugin, PluginDiagnostic,
    PluginError, PluginRequest, PluginResponse, PROTOCOL_VERSION,
};
// Embed plugin protocol — RFC-018. Completely separate from the language Plugin system.
pub use travsr_plugin_protocol::{
    EmbedHandshakeRequest, EmbedHandshakeResponse, EmbedPlugin, EmbedPluginRequest,
    EmbedPluginResponse, EmbedRequest, EmbedResponse, KnnRequest, KnnResponse,
    EMBED_PROTOCOL_VERSION,
};
// Re-export core types so language crates only need travsr-plugin-sdk as a dep.
pub use travsr_core::{Edge, Language, Node, NodeId, VName};

mod runner;
pub use runner::run_plugin;

mod embed_runner;
pub use embed_runner::run_embed_plugin;
