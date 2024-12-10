<script>
    import { onMount, onDestroy } from "svelte";
    import { AudioTranscriptionClient } from "./lib/AudioTranscriptionClient";

    let transcriptionClient;

    let file;
    let readerName = "Lucy";
    let messages = [];
    let userInput = "";
    let isListening = false;
    let bookTitle = "";
    let audioContext;
    let audioInitialized = false;

    onMount(async () => {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        transcriptionClient = new AudioTranscriptionClient(
            "ws://localhost:8765",
        );
    });

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
            await transcriptionClient.startTranscription();

            transcriptionClient.socket.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                console.log("onmessage", data);
                if (
                    data.type === "final_transcript" &&
                    data.is_final &&
                    data.speech_final
                ) {
                    // Update the input field with the transcribed text
                    userInput = data.transcript;

                    // Send the transcribed text to chat
                    // try {
                    //     const response = await fetch(
                    //         "http://localhost:8000/chat",
                    //         {
                    //             method: "POST",
                    //             headers: {
                    //                 "Content-Type": "application/json",
                    //             },
                    //             body: JSON.stringify({
                    //                 message: data.transcript,
                    //                 reader_name: readerName,
                    //             }),
                    //         },
                    //     );

                    //     if (!response.ok) {
                    //         throw new Error("Chat request failed");
                    //     }

                    //     const chatResponse = await response.json();
                    //     messages = [
                    //         ...messages,
                    //         { role: "user", content: data.transcript },
                    //         {
                    //             role: "assistant",
                    //             content: chatResponse.message,
                    //         },
                    //     ];

                    //     if (chatResponse.audio) {
                    //         playAudio(chatResponse.audio);
                    //     }
                    // } catch (error) {
                    //     console.error("Error sending chat message:", error);
                    // }

                    // Stop listening after processing
                    isListening = false;
                    transcriptionClient.stopTranscription();

                    sendMessage();
                } else if (data.type === "interim_transcript") {
                    // Show interim results while speaking
                    userInput = data.transcript;
                }
            };
        } else {
            isListening = false;
            transcriptionClient.stopTranscription();
        }
    }

    onDestroy(() => {
        if (transcriptionClient) {
            transcriptionClient.stopTranscription();
        }
    });

    async function initializeAudioContext() {
        console.log("initializeAudioContext");
        if (!audioInitialized && audioContext?.state === "suspended") {
            await audioContext.resume();
            audioInitialized = true;
        }
    }

    // Modify your existing playAudio function
    async function playAudio(audioData) {
        if (!audioData || !Array.isArray(audioData)) {
            console.log("No valid audio data received");
            return;
        }

        try {
            await initializeAudioContext();

            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(
                1, // mono
                audioData.length,
                48000, // sample rate
            );

            // Get the channel data
            const channelData = audioBuffer.getChannelData(0);

            // Fill the buffer with normalized audio data
            for (let i = 0; i < audioData.length; i++) {
                // Normalize from int16 (-32768 to 32767) to float32 (-1.0 to 1.0)
                channelData[i] = audioData[i] / 32768.0;
            }

            // Create audio source
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(0);

            // For debugging
            console.log("Playing audio:", {
                length: audioData.length,
                sampleRate: audioBuffer.sampleRate,
                duration: audioBuffer.duration,
            });
        } catch (error) {
            console.error("Error playing audio:", error);
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
