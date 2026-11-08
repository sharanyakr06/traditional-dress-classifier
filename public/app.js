const form = document.querySelector("#prediction-form");
const input = document.querySelector("#image-input");
const uploadArea = document.querySelector("#upload-area");
const uploadMain = document.querySelector("#upload-main");
const uploadHint = document.querySelector("#upload-hint");
const previewWrap = document.querySelector("#preview-wrap");
const preview = document.querySelector("#image-preview");
const removeButton = document.querySelector("#remove-image");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = document.querySelector("#button-label");
const statusMessage = document.querySelector("#status-message");
const result = document.querySelector("#result");
const prediction = document.querySelector("#prediction");
let previewUrl = null;

function showStatus(message) {
  statusMessage.textContent = message;
  statusMessage.hidden = !message;
}

function clearResult() {
  result.hidden = true;
}

function useFile(file) {
  clearResult();
  showStatus("");

  if (!file) {
    input.value = "";
    submitButton.disabled = true;
    previewWrap.hidden = true;
    uploadMain.textContent = "Click to choose an image";
    uploadHint.textContent = "or drag and drop it here";
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    return;
  }

  const isSupported = ["image/jpeg", "image/png"].includes(file.type);
  if (!isSupported) {
    input.value = "";
    submitButton.disabled = true;
    previewWrap.hidden = true;
    showStatus("Please choose a JPG, JPEG, or PNG image.");
    return;
  }
  if (file.size > 4 * 1024 * 1024) {
    input.value = "";
    submitButton.disabled = true;
    previewWrap.hidden = true;
    showStatus("The image is too large. Maximum upload size is 4 MB.");
    return;
  }

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  previewWrap.hidden = false;
  uploadMain.textContent = file.name;
  uploadHint.textContent = "Click here to choose a different image";
  submitButton.disabled = false;
}

input.addEventListener("change", () => useFile(input.files[0]));
removeButton.addEventListener("click", () => useFile(null));

for (const eventName of ["dragenter", "dragover"]) {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.remove("is-dragging");
  });
}

uploadArea.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;

  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  useFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) return;

  clearResult();
  showStatus("");
  submitButton.disabled = true;
  buttonLabel.textContent = "Analyzing image...";

  try {
    const formData = new FormData();
    formData.append("image", file);
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(data?.error || "The server could not analyze the image.");
    }
    if (!data) {
      throw new Error("The server returned an invalid response. Please try again.");
    }

    prediction.textContent = data.prediction;
    result.hidden = false;
  } catch (error) {
    showStatus(error.message || "The image could not be analyzed. Please try again.");
  } finally {
    submitButton.disabled = false;
    buttonLabel.textContent = "Analyze image";
  }
});
