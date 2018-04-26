#ifndef ALEX_ASR_FEATURE_PIPELINE_H
#define ALEX_ASR_FEATURE_PIPELINE_H

#include "decoder_config.h"

using namespace kaldi;

namespace alex_asr {
    class FeaturePipeline {
    public:
        FeaturePipeline(DecoderConfig & config);
        ~FeaturePipeline();
        OnlineFeatureInterface *GetFeature();
        OnlineFeatureInterface *GetInputFeature();
        OnlineIvectorFeature* GetIvectorFeature();
        void AcceptWaveform(BaseFloat sampling_rate,
                            const VectorBase<BaseFloat> &waveform);
        void InputFinished();
    private:
        OnlineBaseFeature *base_feature_;
        OnlineCmvn *cmvn_;
        OnlineCmvnState *cmvn_state_;
        OnlineSpliceFrames *splice_;
        OnlineDeltaFeature *delta_;
        OnlineTransform *transform_lda_;
        OnlineIvectorFeature *ivector_;
        OnlineAppendFeature *ivector_append_;
        OnlinePitchFeature *pitch_;
        OnlineProcessPitch *pitch_feature_;
        OnlineAppendFeature *pitch_append_;
        OnlineFeatureInterface *input_feature_;
        OnlineFeatureInterface *final_feature_;
    };
}

#endif //PYKALDI_PYKALDI2_FEATURE_PIPELINE_CC_H
