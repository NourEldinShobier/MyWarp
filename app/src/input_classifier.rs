use std::sync::{Arc, RwLock};

use input_classifier::{HeuristicClassifier, InputClassifier};
use warpui::{Entity, ModelContext, SingletonEntity};

pub struct InputClassifierModel {
    /// Starts as the heuristic classifier and is swapped for the ML classifier once it loads.
    classifier: Arc<RwLock<Arc<dyn InputClassifier>>>,
}

impl InputClassifierModel {
    pub fn new(_ctx: &mut ModelContext<Self>) -> Self {
        let classifier: Arc<RwLock<Arc<dyn InputClassifier>>> =
            Arc::new(RwLock::new(Arc::new(HeuristicClassifier)));

        // Building the model is slow, so do it off the main thread.
        #[cfg(not(target_family = "wasm"))]
        {
            #[cfg(feature = "nld_fasttext_model")]
            let fasttext_enabled = is_nld_classifier_enabled(_ctx);
            #[cfg(not(feature = "nld_fasttext_model"))]
            let fasttext_enabled = false;
            let slot = classifier.clone();
            std::thread::spawn(move || {
                let start = std::time::Instant::now();
                if let Some(loaded) = load_model_classifier(fasttext_enabled) {
                    if let Ok(mut slot) = slot.write() {
                        *slot = loaded;
                    }
                }
                log::info!(
                    "startup perf: input classifier model load took {} ms",
                    start.elapsed().as_millis()
                );
            });
        }

        Self { classifier }
    }

    pub fn classifier(&self) -> Arc<dyn InputClassifier> {
        match self.classifier.read() {
            Ok(classifier) => classifier.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

#[allow(unused_variables)]
fn load_model_classifier(fasttext_enabled: bool) -> Option<Arc<dyn InputClassifier>> {
    #[cfg(feature = "nld_onnx_model")]
    match input_classifier::OnnxClassifier::new(input_classifier::OnnxModel::BertTiny) {
        Ok(classifier) => {
            log::info!("Loaded onnx classifier");
            return Some(Arc::new(classifier));
        }
        Err(e) => log::warn!("Failed to load onnx classifier: {e:#}"),
    }

    #[cfg(feature = "nld_fasttext_model")]
    if fasttext_enabled {
        match input_classifier::FasttextClassifier::new() {
            Ok(classifier) => {
                log::info!("Loaded fasttext classifier");
                return Some(Arc::new(classifier));
            }
            Err(e) => log::warn!("Failed to load fasttext classifier: {e:#}"),
        }
    }

    None
}

impl Entity for InputClassifierModel {
    type Event = ();
}

impl SingletonEntity for InputClassifierModel {}

#[cfg(feature = "nld_fasttext_model")]
/// Returns true iff the NLD classifier model is enabled.
pub fn is_nld_classifier_enabled(ctx: &warpui::AppContext) -> bool {
    use warp_core::user_preferences::GetUserPreferences as _;
    use warp_core::{channel::ChannelState, features::FeatureFlag};

    if ChannelState::channel().is_dogfood() {
        // The `EnableNLDClassifierModel` can be used to force enable / disable
        // use if it is set.
        ctx.private_user_preferences()
            .read_value("EnableNLDClassifierModel")
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok())
            .unwrap_or(FeatureFlag::NLDClassifierModelEnabled.is_enabled())
    } else {
        FeatureFlag::NLDClassifierModelEnabled.is_enabled()
    }
}
