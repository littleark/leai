<script>
    import { onMount } from "svelte";

    let file;
    let readerName = "Lucy";
    let messages = [];
    let userInput = "";
    let isListening = false;
    let bookTitle = "";
    let audioContext;
    let audioInitialized = false;

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    onMount(async () => {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    sampleSize: 16,
                },
            });

            mediaRecorder = new MediaRecorder(stream, {
                mimeType: "audio/webm", // Most browsers support this format
            });

            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                await sendAudioToServer(audioBlob);
            };

            mediaRecorder.start();
            isRecording = true;
        } catch (error) {
            console.error("Error starting recording:", error);
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            mediaRecorder.stream.getTracks().forEach((track) => track.stop());
        }
    }

    async function sendAudioToServer(audioBlob) {
        try {
            // Convert to audio/wav using AudioContext
            const audioContext = new (window.AudioContext ||
                window.webkitAudioContext)();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioData = await audioContext.decodeAudioData(arrayBuffer);

            // Create WAV file
            const wavBuffer = audioBufferToWav(audioData);
            const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

            const formData = new FormData();
            formData.append("audio", wavBlob, "recording.wav");

            const response = await fetch("http://localhost:8000/transcribe", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            userInput = data.transcription;
        } catch (error) {
            console.error("Error sending audio to server:", error);
        }
    }

    // Function to convert AudioBuffer to WAV format
    function audioBufferToWav(audioBuffer) {
        const numChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;

        const buffer = audioBuffer.getChannelData(0);
        const samples = Int16Array.from(buffer.map((n) => n * 0x7fff));
        const dataSize = samples.length * bytesPerSample;

        const headerSize = 44;
        const wavBuffer = new ArrayBuffer(headerSize + dataSize);
        const view = new DataView(wavBuffer);

        // Write WAV header
        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, "data");
        view.setUint32(40, dataSize, true);

        // Write audio data
        for (let i = 0; i < samples.length; i++) {
            view.setInt16(headerSize + i * bytesPerSample, samples[i], true);
        }

        return wavBuffer;
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    async function handleFileUpload() {
        if (!file || !file[0]) {
            console.error("No file selected");
            return;
        }

        const formData = new FormData();
        formData.append("file", file[0]);
        formData.append("reader_name", readerName);

        try {
            const response = await fetch("http://localhost:8000/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Upload failed");
            }

            const data = await response.json();
            bookTitle = data.book_title;
            messages = [
                ...messages,
                { role: "assistant", content: data.welcome_message },
            ];

            // Play welcome audio
            if (data.audio) {
                playAudio(data.audio);
            }
        } catch (error) {
            console.error("Error uploading file:", error);
        }
    }

    async function sendMessage() {
        if (!userInput.trim()) return;

        const message = userInput;
        userInput = "";
        messages = [...messages, { role: "user", content: message }];

        try {
            const response = await fetch("http://localhost:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message,
                    reader_name: readerName,
                }),
            });

            const data = await response.json();
            messages = [
                ...messages,
                { role: "assistant", content: data.message },
            ];

            // Play response audio
            playAudio(data.audio);
        } catch (error) {
            console.error("Error sending message:", error);
        }
    }

    async function toggleListening() {
        if (!isListening) {
            isListening = true;
            await startRecording();
        } else {
            isListening = false;
            stopRecording();
        }
    }

    async function toggleListening2() {
        isListening = !isListening;

        if (isListening) {
            try {
                const response = await fetch(
                    "http://localhost:8000/transcribe",
                    {
                        method: "POST",
                    },
                );

                const data = await response.json();
                userInput = data.transcription;
                isListening = false;
            } catch (error) {
                console.error("Error transcribing:", error);
                isListening = false;
            }
        }
    }

    async function initializeAudioContext() {
        console.log("initializeAudioContext");
        if (!audioInitialized && audioContext?.state === "suspended") {
            await audioContext.resume();
            audioInitialized = true;
        }
    }

    // Modify your existing playAudio function
    async function playAudio(audioData) {
        console.log("playAudio", audioData?.length);
        if (!audioData) return;

        try {
            await initializeAudioContext();

            const sampleRate = 48000;
            const rawSamples = audioData; // Now directly an array of integers

            // Create the audio buffer
            const audioBuffer = audioContext.createBuffer(
                1, // Number of channels
                rawSamples.length, // Number of samples
                sampleRate, // Sample rate
            );

            // Get channel data
            const channelData = audioBuffer.getChannelData(0);

            // Fill the buffer
            for (let i = 0; i < rawSamples.length; i++) {
                // Normalize from int16 (-32768 to 32767) to float32 (-1.0 to 1.0)
                channelData[i] = rawSamples[i] / 32768.0;
            }

            // Create and configure source
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;

            // Create gain node
            const gainNode = audioContext.createGain();
            gainNode.gain.value = 1.0;

            // Connect the nodes
            source.connect(gainNode);
            gainNode.connect(audioContext.destination);

            // Start playing
            source.start(0);

            // Debug logging
            console.log("Audio details:", {
                sampleCount: rawSamples.length,
                duration: audioBuffer.duration,
                sampleRate: audioBuffer.sampleRate,
                maxSample: Math.max(...rawSamples),
                minSample: Math.min(...rawSamples),
            });

            return source;
        } catch (error) {
            console.error("Error playing audio:", error, {
                audioContextState: audioContext?.state,
                dataLength: audioData?.length,
            });
        }
    }
</script>

<svelte:window on:click={initializeAudioContext} />
<main>
    <div class="audio-status" on:click={initializeAudioContext}>
        {#if audioContext}
            <span class={audioInitialized ? "enabled" : "disabled"}>
                🔊 Audio is {audioInitialized ? "enabled" : "disabled"}
            </span>
        {/if}
    </div>

    <h1>{bookTitle || "Book Companion"}</h1>

    <div class="setup">
        <input
            type="text"
            bind:value={readerName}
            placeholder="Reader's name"
        />
        <input
            type="file"
            bind:files={file}
            accept=".pdf,.txt"
            on:change={() => (file = file)}
        />
        <button on:click={handleFileUpload}>Upload Book</button>
    </div>

    <div class="chat">
        {#each messages as message}
            <div class="message {message.role}">
                <p>{message.content}</p>
            </div>
        {/each}
    </div>

    <div class="input-area">
        <input
            type="text"
            bind:value={userInput}
            placeholder="Type your message..."
            on:keypress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button on:click={sendMessage}>Send</button>
        <button on:click={toggleListening}>
            {isListening ? "Stop Listening" : "Start Listening"}
        </button>
    </div>
</main>

<style>
    .chat {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 1rem;
        margin: 1rem 0;
    }

    .message {
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 4px;
        border: 0 solid #ccc;
    }

    .user {
        border-color: #e3f2fd;
        margin-left: 20%;
        text-align: right;
    }

    .assistant {
        border-color: #f5f5f5;
        margin-right: 20%;
        text-align: left;
    }

    .input-area {
        display: flex;
        gap: 0.5rem;
    }

    input[type="text"] {
        flex: 1;
        padding: 0.5rem;
    }

    button {
        padding: 0.5rem 1rem;
    }

    .audio-status {
        position: fixed;
        top: 1rem;
        right: 1rem;
        padding: 0.5rem;
        border-radius: 4px;
        background: #f5f5f5;
    }

    .enabled {
        color: green;
    }

    .disabled {
        color: red;
    }
</style>
