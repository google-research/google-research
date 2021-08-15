"""Video abstraction for rendering and saving rollouts to disk."""

import os
import typing

import imageio

import gym


class VideoRecorder:
    """Record and save policy rollouts to disk."""

    def __init__(
        self,
        root_dir: str,
        resolution: typing.Tuple[int, int] = (256, 256),
        fps: float = 30,
    ) -> None:
        """Constructor.

        Args:
            root_dir: The root directory in which to save the rollout videos.
                If None, recording and saving will be disabled.
            resolution: The (height, width) resolution of the video.
            fps: What frames per second to save videos with.
        """
        self.save_dir = None
        if root_dir:
            self.save_dir = os.path.join(root_dir, "video")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.height, self.width = resolution
        self.fps = fps
        self.enabled = True
        self.frames = []

    def reset(self, enabled: bool = True) -> None:
        """Clear the internal frame buffer.

        Args:
            enabled: When set to False, subsequent calls to `record` and `save`
                are disabled. Note that `root_dir` must not be None for
                enabled=True to work correctly.
        """
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env: gym.Env) -> None:
        """Render the env and store the rendered frame inside a buffer."""
        if self.enabled:
            frame = env.render(mode="rgb_array", height=self.height, width=self.width)
            self.frames.append(frame)

    def save(self, file_name: str) -> None:
        """Save the frame buffer as a video, then clear it."""
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            self.reset()
