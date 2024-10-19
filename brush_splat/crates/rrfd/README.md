## rrfd - really rusty file reader

`rrfd` is `rfd` + an android file picker. It uses some JNI code to start a new activity that returns a file name and the file contents.

This is **not** production ready and just a quick setup to get files working on Android, please use with care.

In the future, hardening this implementation and upstreaming it to `rfd` would be fantastic, but it's likely very hard as there is no universal way to setup the `jni` integration.
