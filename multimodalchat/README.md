# MultimodalChat: suggest rich contents(photos/GIFs/Stickers/...) during conversations
`Photochat/` contains the dataset we release with our [PhotoChat paper]( https://arxiv.org/abs/2108.01453).

Data is formatted in json as following:

```json
{
  "dialogue":[
    {
      "share_photo":"Boolean value denoting whether a photo is shared in this turn.",
      "user_id":"0 or 1. User id of this turn.",
      "message":"Text of one conversation turn. Empty when share_photo is true."
    }
  ],
  "dialogue_id":"Integer. Unique dialogue id.",
  "photo_description":"String. Photo description. It includes info about object labels in the photo.",
  "photo_url":"Photo url."
}
```
Please contact paper authors if you have any questions about the data.
