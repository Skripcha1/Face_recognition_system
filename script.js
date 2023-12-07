const video = document.getElementById("video");
let faceMatcher; // Declare faceMatcher in the outer scope

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
]).then(startWebcam);

function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
}

async function getLabeledFaceDescriptions() {
  const labels = ["Felipe", "Messi", "Data"];
  return Promise.all(
    labels.map(async (label) => {
      try {
        const descriptions = [];
        const imageMap = {};

        for (let i = 1; i <= 2; i++) {
          let img;
          if (!imageMap[label + i]) {
            img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
            imageMap[label + i] = img;
          } else {
            img = imageMap[label + i];
          }

          const detections = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
          descriptions.push(detections.descriptor);
        }

        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      } catch (error) {
        console.error(`Error getting labeled face descriptors for ${label}:`, error);
      }
    })
  );
}

let lastFrameTime = 0;
const displayCanvas = faceapi.createCanvasFromMedia(video);
document.body.append(displayCanvas);

const displaySize = { width: video.width, height: video.height };
faceapi.matchDimensions(displayCanvas, displaySize);

async function drawFaces() {
  const canvas = displayCanvas.getContext("2d");
  canvas.clearRect(0, 0, canvas.width, canvas.height);

  const detections = await faceapi
    .detectAllFaces(video, new faceapi.SsdMobilenetv1Options())
    .withFaceDescriptors();

  const resizedDetections = faceapi.resizeResults(detections, {
    height: video.height,
    width: video.width,
  });

  const results = resizedDetections.map((d) => {
    return faceMatcher.findBestMatch(d.descriptor);
  });

  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, {
      label: result,
    });
    drawBox.draw(canvas);
  });

  requestAnimationFrame(drawFaces);
}

video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  drawFaces();
});
