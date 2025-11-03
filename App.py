
import gradio as gr
from Inference import predict_from_bytes

def classify_image(img):
    # gradio gives PIL images; convert to bytes if your inference expects bytes
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = predict_from_bytes(buf.getvalue())
    return f"Label: {result['label']}", f"Confidence: {result['confidence']:.3f}"

title = "Fake Image Detector"
desc = "Detects whether an image is synthetic/manipulated."

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Confidence")],
    title=title,
    description=desc,
    live=False,
)

if __name__ == "__main__":
    demo.launch()
