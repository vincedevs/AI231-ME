# Wayne Manor audio assets

The user-supplied telephone ringing sound is stored here as
`telephone-ring.wav`. `telephone.ringtone_url` in
`wayne_manor/config/default.json` points to its public path:

```json
"ringtone_url": "/assets/telephone-ring.wav"
```

Rebuild the frontend after replacing the file. The UI always shows and animates
the ringing telephone. Audio requires one browser interaction through
**Enable ringtone audio** because browsers normally block unsolicited autoplay.
