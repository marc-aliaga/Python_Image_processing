const form = document.getElementById("form");
const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropText = document.getElementById("drop-text");
const submitBtn = document.getElementById("submit");
const statusEl = document.getElementById("status");
const inputImg = document.getElementById("input-img");
const outputImg = document.getElementById("output-img");
const download = document.getElementById("download");

let selectedFile = null;

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Selecciona un archivo de imagen.", true);
    return;
  }
  selectedFile = file;
  dropText.textContent = file.name;
  submitBtn.disabled = false;
  if (inputImg.src) URL.revokeObjectURL(inputImg.src);
  inputImg.src = URL.createObjectURL(file);
  outputImg.removeAttribute("src");
  download.hidden = true;
  setStatus("");
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
);
drop.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  const data = new FormData();
  data.append("image", selectedFile);

  submitBtn.disabled = true;
  setStatus("Procesando…");
  try {
    const res = await fetch("/process", { method: "POST", body: data });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Error ${res.status}`);
    }
    const blob = await res.blob();
    if (outputImg.src) URL.revokeObjectURL(outputImg.src);
    const url = URL.createObjectURL(blob);
    outputImg.src = url;
    download.href = url;
    download.hidden = false;
    setStatus("Listo.");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    submitBtn.disabled = false;
  }
});
