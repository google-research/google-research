
Examples of input/output audio data generated with models from paper [Streaming Parrotron for on-device speech-to-speech conversion](https://arxiv.org/abs/2210.13761).


VCTK data [license](https://datashare.ed.ac.uk/bitstream/handle/10283/3443/license_text?sequence=3&isAllowed=y)
|  Audio files      | Description  |
| ---------------- | --------------------- |
|[Input](vctk/input) | Input data containing clean audio clips from VCTK data.     |
|[nEnc_nDec_nGL](vctk/nEnc_nDec_nGL) | Model with non streaming encoder, non streaming decoder and non streaming Griffin Lim.  |
|[LSA LS2(int8),sDec(int8)+GL](vctk/stR764CR44_int8_sDec_int8_sGL1) | Model with quantized int8 streaming encoder LSA_LS2, quantized int8 streaming decoder and streaming vocoder Griffin Lim sGL1 (in this paper, it corresponds to audio examples Table 5: LSA LS2 int8, sDec int8 with GL).  |
|[LSA_LS2(int8)+sDec(int8)+MG](vctk/stR764CR44_int8_sDec_int8_sMelGan1) | Model with quantized int8 streaming encoder LSA_LS2, quantized int8 streaming decoder and streaming neural vocoder sMelGan1 (in this paper, it corresponds to audio examples from Table 5: LSA LS2 int8, sDec int8 with MG).  |

We preserved audio clip names from VCTK dataset.
