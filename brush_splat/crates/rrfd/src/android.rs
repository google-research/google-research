use super::PickedFile;
use anyhow::Result;
use async_channel::Sender;
use jni::objects::{GlobalRef, JByteArray, JClass, JStaticMethodID, JString};
use jni::signature::Primitive;
use jni::JNIEnv;
use lazy_static::lazy_static;
use std::sync::Arc;
use std::sync::RwLock;

lazy_static! {
    static ref VM: RwLock<Option<Arc<jni::JavaVM>>> = RwLock::new(None);
    static ref CHANNEL: RwLock<Option<Sender<Result<PickedFile>>>> = RwLock::new(None);
    static ref START_FILE_PICKER: RwLock<Option<JStaticMethodID>> = RwLock::new(None);
    static ref FILE_PICKER_CLASS: RwLock<Option<GlobalRef>> = RwLock::new(None);
}

pub fn jni_initialize(vm: Arc<jni::JavaVM>) {
    let mut env = vm.get_env().expect("Cannot get reference to the JNIEnv");
    let class = env.find_class("com/splats/app/FilePicker").unwrap();
    let method = env
        .get_static_method_id(&class, "startFilePicker", "()V")
        .unwrap();
    *FILE_PICKER_CLASS.write().expect("Failed to write JNI data.") = Some(env.new_global_ref(class).unwrap());
    *START_FILE_PICKER.write().expect("Failed to write JNI data.") = Some(method);
    *VM.write().unwrap() = Some(vm);
}

pub(crate) async fn pick_file() -> Result<PickedFile> {
    let (sender, receiver) = async_channel::bounded(1);
    {
        let channel = CHANNEL.write();
        if let Ok(mut channel) = channel {
            *channel = Some(sender);
        } else {
            anyhow::bail!("Failed to initialize file picker");
        }
    }

    // Call method. Be sure this is scoped so we drop all guards before waiting.
    {
        let java_vm = VM.read().unwrap().clone().expect("Failed to initialize Java VM");
        let mut env = java_vm.attach_current_thread()?;

        let class = FILE_PICKER_CLASS.read().expect("Failed to initialize FilePicker class");
        let method = START_FILE_PICKER.read().expect("Failed to initialize FilePicker method");
        
        // SAFETY: This is safe as long as we cached the method in the right way, and
        // this matches the Java side. Not much more we can do here.
        let _ = unsafe {
            env.call_static_method_unchecked(
                class.as_ref().expect("Failed to get class reference"),
                method.as_ref().expect("Failed to get method reference"),
                jni::signature::ReturnType::Primitive(Primitive::Void),
                &[],
            )
        }?;
    }

    receiver.recv().await?
}

#[no_mangle]
extern "system" fn Java_com_splats_app_FilePicker_onFilePickerResult<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    data: JByteArray<'local>,
    file_name: JString<'local>,
) {
    // Channel can be gone before the callback if other parts of pick_file fail.
    if let Ok(ch) = CHANNEL.read() {
        if let Some(ch) = ch.as_ref() {
            let picked_file = if data.is_null() {
                Err(jni::errors::Error::NullPtr("No file selected"))
            } else {
                env.convert_byte_array(data).and_then(|data| {
                    let file_name = env.get_string(&file_name)?.into();
                    Ok(PickedFile { data, file_name })
                })
            };
            ch.try_send(picked_file.map_err(|err| err.into())).expect("Failed to send file picking result");
        }
    }
}
