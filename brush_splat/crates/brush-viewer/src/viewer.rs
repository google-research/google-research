use anyhow::Context;
use async_channel::{Receiver, Sender, TryRecvError, TrySendError};
use brush_dataset::scene_batch::SceneLoader;
use brush_render::{camera::Camera, gaussian_splats::Splats};
use burn::tensor::ElementConversion;
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::{JitBackend, RuntimeOptions, WgpuDevice, WgpuRuntime};
use egui::{pos2, CollapsingHeader, Color32, Hyperlink, Rect, Slider};
use futures_lite::StreamExt;
use glam::{Quat, Vec2, Vec3};

use tracing::info_span;
use web_time::Instant;
use wgpu::CommandEncoderDescriptor;

use brush_dataset;
use brush_train::scene::Scene;
use brush_train::train::{LrConfig, SplatTrainer, TrainConfig};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use crate::{burn_texture::BurnTexture, orbit_controls::OrbitControls, splat_import};

type Backend = JitBackend<WgpuRuntime, f32, i32>;

enum ViewerMessage {
    Initial,
    Error(anyhow::Error),
    SplatLoad {
        splats: Splats<Backend>,
        total_count: usize,
    },
    TrainStep {
        splats: Splats<Backend>,
        loss: f32,
        iter: u32,
        timestamp: Instant,
    },
}

pub struct Viewer {
    receiver: Option<Receiver<ViewerMessage>>,
    last_message: Option<ViewerMessage>,

    last_train_iter: u32,
    ctx: egui::Context,

    device: WgpuDevice,

    splat_view: SplatView,

    train_iter_per_s: f32,
    last_train_step: (Instant, u32),
    train_pause: bool,

    file_path: String,

    target_train_resolution: u32,
    max_frames: usize,

    constant_redraww: bool,
}

struct SplatView {
    camera: Camera,
    controls: OrbitControls,
    backbuffer: Option<BurnTexture>,
    last_draw: Instant,
    sync_render: bool,
}

struct TrainArgs {
    frame_count: usize,
    target_resolution: u32,
}

async fn load_ply_loop(
    data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
) -> anyhow::Result<()> {
    let total_count = splat_import::ply_count(data).context("Invalid ply file")?;

    let splat_stream = splat_import::load_splat_from_ply::<Backend>(data, device.clone());

    let mut splat_stream = std::pin::pin!(splat_stream);
    while let Some(splats) = splat_stream.next().await {
        egui_ctx.request_repaint();

        let splats = splats?;
        let msg = ViewerMessage::SplatLoad {
            splats,
            total_count,
        };

        sender
            .send(msg)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))?;
    }

    Ok(())
}

async fn train_loop(
    data: &[u8],
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
) -> anyhow::Result<()> {
    let config = TrainConfig::new(
        LrConfig::new().with_max_lr(4e-5).with_min_lr(2e-5),
        LrConfig::new().with_max_lr(8e-2).with_min_lr(2e-2),
        LrConfig::new().with_max_lr(2e-2).with_min_lr(1e-2),
    );

    let cameras = brush_dataset::read_dataset(
        data,
        Some(train_args.frame_count),
        Some(train_args.target_resolution),
    )?;
    let scene = Scene::new(cameras);

    #[cfg(feature = "rerun")]
    let visualize = crate::visualize::VisualizeTools::new();

    #[cfg(feature = "rerun")]
    visualize.log_scene(&scene)?;

    let mut splats =
        Splats::<Autodiff<Backend>>::init_random(config.init_splat_count, 2.0, &device);

    let mut dataloader = SceneLoader::new(scene, &device, 1);
    let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

    loop {
        let batch = {
            let _span = info_span!("Get batch").entered();
            dataloader.next_batch()
        };

        #[cfg(feature = "rerun")]
        let gt_image = batch.gt_images.clone();

        let (new_splats, stats) = trainer.step(batch, splats).await.unwrap();

        #[cfg(feature = "rerun")]
        {
            if trainer.iter % config.visualize_splats_every == 0 {
                visualize
                    .log_train_stats(&new_splats, &stats, gt_image)
                    .await?;
            }

            if trainer.iter % config.visualize_splats_every == 0 {
                visualize.log_splats(&new_splats).await?;
            }
        }

        splats = new_splats;

        if trainer.iter % 5 == 0 {
            let _span = info_span!("Send batch").entered();
            let msg = ViewerMessage::TrainStep {
                splats: splats.valid(),
                loss: stats.loss.into_scalar_async().await.elem::<f32>(),
                iter: trainer.iter,
                timestamp: Instant::now(),
            };

            match sender.try_send(msg) {
                Ok(_) => (),
                Err(TrySendError::Full(_)) => (),
                Err(_) => {
                    break; // channel closed, bail.
                }
            }
            egui_ctx.request_repaint();
        }

        // On wasm, yield to the browser.
        #[cfg(target_arch = "wasm32")]
        gloo_timers::future::TimeoutFuture::new(0).await;
    }

    Ok(())
}

async fn process_loop(
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    egui_ctx: egui::Context,
    train_args: TrainArgs,
) -> anyhow::Result<()> {
    let picked = rrfd::pick_file().await?;

    if picked.file_name.contains(".ply") {
        load_ply_loop(&picked.data, device, sender, egui_ctx).await
    } else if picked.file_name.contains(".zip") {
        train_loop(&picked.data, device, sender, egui_ctx, train_args).await
    } else {
        anyhow::bail!("Only .ply and .zip files are supported.")
    }
}

impl SplatView {
    fn draw_splats(
        &mut self,
        splats: &Splats<Backend>,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        CollapsingHeader::new("View Splats")
            .default_open(true)
            .show(ui, |ui| {
                let size = ui.available_size();

                // Round to 64 pixels for buffer alignment.
                let size =
                    glam::uvec2(((size.x as u32 / 64) * 64).max(32), (size.y as u32).max(32));

                let (rect, response) = ui.allocate_exact_size(
                    egui::Vec2::new(size.x as f32, size.y as f32),
                    egui::Sense::drag(),
                );

                let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);

                let (pan, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
                    (Vec2::ZERO, mouse_delta)
                } else if response.dragged_by(egui::PointerButton::Secondary)
                    || response.dragged_by(egui::PointerButton::Middle)
                {
                    (mouse_delta, Vec2::ZERO)
                } else {
                    (Vec2::ZERO, Vec2::ZERO)
                };

                let scrolled = ui.input(|r| r.smooth_scroll_delta).y;
                let cur_time = Instant::now();
                let delta_time = cur_time - self.last_draw;
                self.last_draw = cur_time;

                self.controls.pan_orbit_camera(
                    &mut self.camera,
                    pan * 5.0,
                    rotate * 5.0,
                    scrolled * 0.01,
                    glam::vec2(rect.size().x, rect.size().y),
                    delta_time.as_secs_f32(),
                );

                let base_fov = 0.75;
                self.camera.fov =
                    glam::vec2(base_fov, base_fov * (size.y as f32) / (size.x as f32));

                // If there's actual rendering to do, not just an imgui update.
                if ctx.has_requested_repaint() {
                    if self.sync_render {
                        sync_span::set_enabled(true);
                        info_span!("Pre render", sync_burn = true).in_scope(|| {});
                        sync_span::set_enabled(false);
                    }
                    let _span = info_span!("Render splats").entered();
                    let (img, _) =
                        splats.render(&self.camera, size, glam::vec3(0.0, 0.0, 0.0), true);

                    if self.sync_render {
                        sync_span::set_enabled(true);
                        info_span!("Post render", sync_burn = true).in_scope(|| {});
                        sync_span::set_enabled(false);
                    }

                    let back = self
                        .backbuffer
                        .get_or_insert_with(|| BurnTexture::new(img.clone(), frame));

                    {
                        let state = frame.wgpu_render_state();
                        let state = state.as_ref().unwrap();
                        let mut encoder =
                            state
                                .device
                                .create_command_encoder(&CommandEncoderDescriptor {
                                    label: Some("viewer encoder"),
                                });
                        back.update_texture(img, frame, &mut encoder);
                        let cmd = encoder.finish();
                        state.queue.submit([cmd]);
                    }
                }

                if let Some(back) = self.backbuffer.as_ref() {
                    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                    ui.painter().image(
                        back.id,
                        rect,
                        Rect {
                            min: pos2(0.0, 0.0),
                            max: pos2(1.0, 1.0),
                        },
                        Color32::WHITE,
                    );
                }
            });
    }
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();

        // Run the burn backend on the egui WGPU device.
        let device = burn::backend::wgpu::init_existing_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
            // Splatting workload is much more granular, so don't want to flush as often.
            RuntimeOptions { tasks_max: 64 },
        );

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                use tracing_subscriber::layer::SubscriberExt;

                let subscriber = tracing_subscriber::registry().with(tracing_wasm::WASMLayer::new(Default::default()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            } else if #[cfg(feature = "tracy")] {
                use tracing_subscriber::layer::SubscriberExt;
                let subscriber = tracing_subscriber::registry()
                    .with(tracing_tracy::TracyLayer::default())
                    .with(sync_span::SyncLayer::new(device.clone()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            }
        }

        Viewer {
            receiver: None,
            last_message: None,
            last_train_iter: 0,
            ctx: cc.egui_ctx.clone(),
            splat_view: SplatView {
                camera: Camera::new(
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    glam::vec2(0.5, 0.5),
                    glam::vec2(0.5, 0.5),
                ),
                backbuffer: None,
                controls: OrbitControls::new(7.0),
                last_draw: Instant::now(),
                sync_render: false,
            },
            train_pause: false,
            train_iter_per_s: 0.0,
            last_train_step: (Instant::now(), 0),
            device,
            file_path: "/path/to/file".to_string(),
            target_train_resolution: 800,
            max_frames: 32,
            constant_redraww: false,
        }
    }

    pub fn start_data_load(&mut self) {
        <Backend as burn::prelude::Backend>::seed(42);

        // create a channel for the train loop.
        let (sender, receiver) = async_channel::bounded(2);
        let device = self.device.clone();
        self.receiver = Some(receiver);
        let ctx = self.ctx.clone();

        // Reset camera & controls.
        self.splat_view.camera = Camera::new(
            Vec3::ZERO,
            Quat::from_rotation_y(std::f32::consts::PI / 2.0)
                * Quat::from_rotation_x(-std::f32::consts::PI / 8.0),
            glam::vec2(0.5, 0.5),
            glam::vec2(0.5, 0.5),
        );
        self.splat_view.controls = OrbitControls::new(7.0);
        self.train_pause = false;

        async fn inner_process_loop(
            device: WgpuDevice,
            sender: Sender<ViewerMessage>,
            egui_ctx: egui::Context,
            train_args: TrainArgs,
        ) {
            match process_loop(device, sender.clone(), egui_ctx, train_args).await {
                Ok(_) => (),
                Err(e) => {
                    let _ = sender.send(ViewerMessage::Error(e)).await;
                }
            }
        }

        let train_args = TrainArgs {
            frame_count: self.max_frames,
            target_resolution: self.target_train_resolution,
        };

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            pollster::block_on(inner_process_loop(device, sender, ctx, train_args))
        });

        #[cfg(target_arch = "wasm32")]
        spawn_local(inner_process_loop(device, sender, ctx, train_args));
    }

    fn url_button(&mut self, label: &str, url: &str, ui: &mut egui::Ui) {
        ui.add(Hyperlink::from_label_and_url(label, url).open_in_new_tab(true));
    }

    fn tracy_debug_ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.constant_redraww, "Constant redraw");
        ui.checkbox(&mut self.splat_view.sync_render, "Sync post render");

        let mut checked = sync_span::is_enabled();
        ui.checkbox(&mut checked, "Sync scopes");
        sync_span::set_enabled(checked);
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.constant_redraww {
            ctx.request_repaint();
        }

        let _span = info_span!("Draw UI").entered();

        egui::Window::new("Load data")
            .anchor(egui::Align2::RIGHT_TOP, (0.0, 0.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Select a .ply to visualize, or a .zip with training data.");

                    ui.add_space(15.0);

                    if ui.button("Pick a file").clicked() {
                        self.start_data_load();
                    }
                });

                ui.add_space(10.0);

                ui.heading("Train settings");
                ui.add(
                    Slider::new(&mut self.target_train_resolution, 32..=2048)
                        .text("Target train resolution"),
                );
                ui.add(Slider::new(&mut self.max_frames, 1..=256).text("Max frames"));

                ui.add_space(15.0);

                if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(25.0);

            if let Some(rx) = self.receiver.as_mut() {
                if !self.train_pause {
                    match rx.try_recv() {
                        Ok(message) => {
                            if let ViewerMessage::TrainStep {
                                iter, timestamp, ..
                            } = &message
                            {
                                self.train_iter_per_s = (iter - self.last_train_step.1) as f32
                                    / (*timestamp - self.last_train_step.0).as_secs_f32();
                                self.last_train_step = (*timestamp, *iter);
                            };

                            self.last_message = Some(message)
                        }
                        Err(TryRecvError::Empty) => (), // nothing to do.
                        Err(TryRecvError::Closed) => self.receiver = None, // channel closed.
                    }
                }
            }

            #[cfg(feature = "tracy")]
            self.tracy_debug_ui(ui);

            if let Some(message) = self.last_message.as_ref() {
                match message {
                    ViewerMessage::Initial => (),
                    ViewerMessage::Error(e) => {
                        ui.label("Error: ".to_owned() + &e.to_string());
                    }
                    ViewerMessage::SplatLoad {
                        splats,
                        total_count,
                    } => {
                        ui.horizontal(|ui| {
                            ui.label(format!("{} splats", splats.num_splats()));

                            if splats.num_splats() < *total_count {
                                ui.label(format!(
                                    "Loading... ({}%)",
                                    splats.num_splats() as f32 / *total_count as f32 * 100.0
                                ));
                            }
                        });
                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
                    ViewerMessage::TrainStep {
                        splats,
                        loss,
                        iter,
                        timestamp: _,
                    } => {
                        ui.horizontal(|ui| {
                            ui.label(format!("{} splats", splats.num_splats()));
                            ui.label(format!(
                                "Train step {iter}, {:.1} steps/s",
                                self.train_iter_per_s
                            ));

                            ui.label(format!("loss: {loss:.3e}"));

                            let paused = self.train_pause;
                            ui.toggle_value(&mut self.train_pause, if paused { "⏵" } else { "⏸" });
                        });

                        self.splat_view.draw_splats(splats, ui, ctx, frame);
                    }
                }
            } else if self.receiver.is_some() {
                ui.label("Loading...");
            }
        });

        if self.splat_view.controls.is_animating() {
            ctx.request_repaint();
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}
