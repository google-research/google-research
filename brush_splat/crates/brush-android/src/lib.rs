#![cfg(target_os = "android")]
use jni::sys::{jint, JNI_VERSION_1_6};
use std::os::raw::c_void;
use std::sync::Arc;

#[allow(non_snake_case)]
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _: *mut c_void) -> jint {
    let vm_ref = Arc::new(vm);
    rrfd::android::jni_initialize(vm_ref);
    JNI_VERSION_1_6
}

#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let wgpu_options = brush_viewer::wgpu_config::get_config();

    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    eframe::run_native(
        "Brush 🖌️",
        eframe::NativeOptions {
            event_loop_builder: Some(Box::new(|builder| {
                builder.with_android_app(app);
            })),
            wgpu_options,
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(brush_viewer::viewer::Viewer::new(cc)))),
    )
    .unwrap();
}
