

def test_output_video_path_auto_is_named_after_the_clip():
    from app.output.recorder import resolve_output_path

    assert resolve_output_path("auto", "data/input/clip1.mp4").replace("\\", "/") == "outputs/clip1_annotated.mp4"
    assert resolve_output_path("auto", "rtsp://cam/stream", "gate").replace("\\", "/") == "outputs/gate_live.mp4"
    assert resolve_output_path("auto", 0, "gate").replace("\\", "/") == "outputs/gate_live.mp4"
    assert resolve_output_path("runs/x.mp4", "a.mp4") == "runs/x.mp4"
    assert resolve_output_path("", "a.mp4") == "" and resolve_output_path(None, "a.mp4") == ""
