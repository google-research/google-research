# TODO update paper name and add link to the paper
Examples of input/output audio data generated with models from paper [PARROTRON: ON-DEVICE STREAMING SPEECH TO SPEECH CONVERSION](TBD).


VCTK data [license](https://datashare.ed.ac.uk/bitstream/handle/10283/3443/license_text?sequence=3&isAllowed=y)
|  Audio files      | Description  |
| ---------------- | --------------------- |
|[Input](vctk/input) | Input data containing clean audio clips from VCTK data.     |
|[nEnc_nDec_nGL](vctk/nEnc_nDec_nGL) | Model with non streaming encoder, non streaming decoder and non streaming Griffin Lim.  |
|[stR764CR44_int8_sDec_int8_sGL1](vctk/stR764CR44_int8_sDec_int8_sGL1) | Model with quantized int8 streaming encoder stR764CR44, quantized int8 streaming decoder and streaming vocoder Griffin Lim sGL1.  |
|[stR764CR44_int8_sDec_int8_sMelGan1](vctk/stR764CR44_int8_sDec_int8_sMelGan1) | Model with quantized int8 streaming encoder stR764CR44, quantized int8 streaming decoder and streaming neural vocoder sMelGan1.  |

We preserved audio clip names from VCTK dataset.
