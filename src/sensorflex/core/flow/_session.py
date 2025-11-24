"""A library for session management."""


class Session:
    class SourceFrame:
        pass

    class TargetFrame:
        pass

    def __init__(self) -> None:
        pass

    def get_new_source_frame(self, frame_index: int):
        pass

    def forward(self, frame: SourceFrame) -> TargetFrame:
        raise NotImplementedError("Not implemented yet.")

    def on_source_frame_loaded(self, source_frame: SourceFrame, frame_index: int):
        pass

    def on_target_frame_created(self, target_frame: TargetFrame, frame_index: int):
        pass
