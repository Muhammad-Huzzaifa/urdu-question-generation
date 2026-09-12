const source = document.querySelector('#source');
const markButton = document.querySelector('#mark');
const generateButton = document.querySelector('#generate');
const status = document.querySelector('#status');
const greedy = document.querySelector('#greedy');
const beam = document.querySelector('#beam');

markButton.addEventListener('click', () => {
    const start = source.selectionStart;
    const end = source.selectionEnd;
    if (start === end) {
        status.textContent = 'Select the answer text first.';
        return;
    }
    const selected = source.value.slice(start, end);
    source.setRangeText(`<ans> ${selected} </ans>`, start, end, 'end');
    status.textContent = 'Answer marked.';
});

generateButton.addEventListener('click', async () => {
    status.textContent = 'Generating questions...';
    generateButton.disabled = true;
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: source.value })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'The request could not be completed.');
        greedy.textContent = data.greedy || 'کوئی نتیجہ نہیں ملا۔';
        beam.textContent = data.beam || 'کوئی نتیجہ نہیں ملا۔';
        status.textContent = 'Complete.';
    } catch (error) {
        status.textContent = error.message;
    } finally {
        generateButton.disabled = false;
    }
});