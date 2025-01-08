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

    let availableBooks = [];
    let selectedBook = "";
    let isLoadingBook = false;

    $: console.log("availableBooks", availableBooks);
    $: console.log("selectedBook", selectedBook);

    let showUploadModal = !bookTitle;
    let isUploading = false;

    const loadingMessage = {
        role: "assistant",
        content: "...",
        isLoading: true,
    };

    const URL = "https://littlebeez-book-companion.hf.space";
    // const URL = "http://localhost:7860";

    onMount(async () => {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        transcriptionClient = new AudioTranscriptionClient(
            // `ws://localhost:7860/ws`,
            `wss://littlebeez-book-companion.hf.space/ws`,
        );

        await fetchAvailableBooks();
    });

    async function fetchAvailableBooks() {
        try {
            const response = await fetch(`${URL}/books`);
            const data = await response.json();
            availableBooks = data.books || [];
        } catch (error) {
            console.error("Error fetching books:", error);
        }
    }

    async function loadBook(collectionName) {
        try {
            const response = await fetch(
                `${URL}/books/${collectionName}/load`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ reader_name: readerName }),
                },
            );

            if (!response.ok) {
                throw new Error("Failed to load book");
            }

            const data = await response.json();
            bookTitle = data.book_title;
            messages = [
                ...messages,
                { role: "assistant", content: data.welcome_message },
            ];
            if (data.audio) {
                playAudio(data.audio);
            }
            showUploadModal = false;
        } catch (error) {
            console.error("Error loading book:", error);
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
            const response = await fetch(`${URL}/upload`, {
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

            if (data.audio) {
                playAudio(data.audio);
            }
        } catch (error) {
            console.error("Error uploading file:", error);
            throw error; // Re-throw to handle in the calling function
        }
    }

    async function sendMessage() {
        if (!userInput.trim()) return;

        const message = userInput;
        userInput = "";
        messages = [...messages, { role: "user", content: message }];
        // Add loading message
        messages = [...messages, loadingMessage];

        try {
            const response = await fetch(`${URL}/chat`, {
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
            messages = messages
                .slice(0, -1)
                .concat([{ role: "assistant", content: data.message }]);

            // Play response audio
            playAudio(data.audio);
        } catch (error) {
            console.error("Error sending message:", error);
            // Remove loading message if there's an error
            messages = messages.slice(0, -1);
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

    async function toggleListening2() {
        if (!isListening) {
            console.log("Starting transcription...");
            isListening = true;
            console.log("STARTING TRANSCRIPTION");
            await transcriptionClient.startTranscription();
            console.log("STARTED TRANSCRIPTION");
            // transcriptionClient.socket.onopen = () => {
            //     console.log("WebSocket connected");
            // };

            // transcriptionClient.socket.onerror = (error) => {
            //     console.error("WebSocket error:", error);
            // };

            // transcriptionClient.socket.onclose = () => {
            //     console.log("WebSocket closed");
            // };

            // transcriptionClient.socket.onmessage = async (event) => {
            //     const data = JSON.parse(event.data);
            //     console.log("Received transcription:", data);
            //     // ... rest of your onmessage handler
            // };
        } else {
            console.log("Stopping transcription...");
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
<main class="container">
    <div class="audio-status" on:click={initializeAudioContext}>
        {#if audioContext}
            <span class={audioInitialized ? "enabled" : "disabled"}>
                {audioInitialized
                    ? "🔊 Ready to chat!"
                    : "🔇 Click to enable audio"}
            </span>
        {/if}
    </div>

    <header>
        <h1>{bookTitle || "📚 Book Companion"}</h1>
        <p class="subtitle">Your friendly reading discussion partner</p>

        {#if !bookTitle}
            <button
                class="upload-btn"
                on:click={() => (showUploadModal = true)}
            >
                📚 Upload a Book to Start
            </button>
        {/if}
    </header>

    {#if showUploadModal}
        <div
            class="modal-backdrop"
            on:click|self={() => !isUploading && (showUploadModal = false)}
        >
            <div class="modal">
                {#if isUploading}
                    <div class="loading-container">
                        <div class="spinner"></div>
                        <h2>Uploading your book...</h2>
                        <p>Please wait while I prepare for our discussion</p>
                    </div>
                {:else}
                    <h2>Let's Start Reading Together! 📚</h2>
                    <div class="modal-content">
                        <div class="setup-item">
                            <label for="reader-name">What's your name?</label>
                            <input
                                id="reader-name"
                                type="text"
                                bind:value={readerName}
                                placeholder="Enter your name"
                            />
                        </div>

                        <!-- Add this new section for book selection -->
                        {#if availableBooks.length > 0}
                            <div class="setup-item">
                                <label for="book-select"
                                    >Select an existing book</label
                                >
                                <select
                                    id="book-select"
                                    bind:value={selectedBook}
                                    class="book-select"
                                    on:change={() =>
                                        console.log(
                                            "Selected book:",
                                            selectedBook,
                                        )}
                                >
                                    <option value=""
                                        >Select a book to read together</option
                                    >
                                    {#each availableBooks as book}
                                        <option value={book.collection_name}>
                                            {book.title}
                                        </option>
                                    {/each}
                                </select>
                                {#if selectedBook}
                                    <button
                                        class="load-btn {isLoadingBook
                                            ? 'loading'
                                            : ''}"
                                        on:click={async () => {
                                            isLoadingBook = true;
                                            try {
                                                await loadBook(selectedBook);
                                            } catch (error) {
                                                console.error(
                                                    "Error loading book:",
                                                    error,
                                                );
                                            } finally {
                                                isLoadingBook = false;
                                            }
                                        }}
                                        disabled={isLoadingBook}
                                    >
                                        {#if isLoadingBook}
                                            <div class="button-spinner"></div>
                                            Loading...
                                        {:else}
                                            Load Selected Book
                                        {/if}
                                    </button>
                                {/if}
                            </div>
                            <div class="setup-item divider">
                                <span>OR</span>
                            </div>
                        {/if}

                        <div class="setup-item">
                            <label for="book-upload">Upload a new book</label>
                            <div class="file-upload">
                                <input
                                    id="book-upload"
                                    type="file"
                                    bind:files={file}
                                    accept=".pdf,.txt"
                                    on:change={() => (file = file)}
                                />
                            </div>
                        </div>
                        <div class="modal-actions">
                            <button
                                class="cancel-btn"
                                on:click={() => (showUploadModal = false)}
                            >
                                Cancel
                            </button>
                            {#if file && file[0]}
                                <button
                                    class="upload-btn"
                                    on:click={async () => {
                                        isUploading = true;
                                        try {
                                            await handleFileUpload();
                                            await fetchAvailableBooks(); // Refresh the book list
                                            showUploadModal = false;
                                        } catch (error) {
                                            console.error(
                                                "Upload failed:",
                                                error,
                                            );
                                        } finally {
                                            isUploading = false;
                                        }
                                    }}
                                >
                                    Start Reading Together
                                </button>
                            {/if}
                        </div>
                    </div>
                {/if}
            </div>
        </div>
    {/if}

    <div class="chat-container">
        {#each [...messages].reverse() as message}
            <div class="message {message.role}">
                <div class="message-content">
                    <span class="avatar">
                        {message.role === "user" ? "👤" : "🤖"}
                    </span>
                    {#if message.isLoading}
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    {:else}
                        <p>{message.content}</p>
                    {/if}
                </div>
            </div>
        {/each}
    </div>

    <div class="input-container">
        <input
            type="text"
            bind:value={userInput}
            placeholder="Share your thoughts..."
            on:keypress={(e) => e.key === "Enter" && sendMessage()}
        />
        <div class="button-group">
            <button class="send-btn" on:click={sendMessage}> Send 📨 </button>
            <button
                class="mic-btn {isListening ? 'recording' : ''}"
                on:click={toggleListening}
            >
                {isListening ? "🎤 Recording..." : "🎤 Speak"}
            </button>
        </div>
    </div>
</main>

<style>
    .container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    header {
        text-align: center;
        margin-bottom: 2rem;
    }

    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
    }

    .setup-container {
        background: #f8f9fa;
        border-radius: 12px;
        margin-bottom: 2rem;
    }

    .setup-item {
        margin-bottom: 1rem;
    }

    .setup-item label {
        display: block;
        margin-bottom: 0.5rem;
        color: #34495e;
        font-weight: 500;
    }

    .file-upload {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    input[type="text"] {
        padding: 0.8rem;
        border: 2px solid #bdc3c7;
        border-radius: 8px;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    input[type="text"]:focus {
        border-color: #3498db;
        outline: none;
    }
    input[type="file"] {
        padding: 0.8rem;
        border: 2px solid #bdc3c7;
        border-radius: 8px;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        margin: 1rem 0;
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column-reverse;
    }

    .message {
        margin: 1rem 0;
    }

    .message-content {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        padding: 1rem;
        border-radius: 12px;
        max-width: 80%;
    }

    .avatar {
        font-size: 1.5rem;
    }

    .user .message-content {
        margin-left: auto;
        background: #3498db;
        color: white;
    }

    .assistant .message-content {
        margin-right: auto;
        background: #f0f2f5;
        color: #2c3e50;
    }

    .input-container {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        margin-top: 1rem;
    }
    .input-container input[type="text"] {
        flex-grow: 1;
    }

    .button-group {
        display: flex;
        gap: 0.5rem;
    }

    button {
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .upload-btn {
        background: #27ae60;
        color: white;
    }

    .send-btn {
        background: #3498db;
        color: white;
    }

    .mic-btn {
        background: #e74c3c;
        color: white;
    }

    .mic-btn.recording {
        background: #c0392b;
        animation: pulse 1.5s infinite;
    }

    button:hover {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    .audio-status {
        position: fixed;
        display: none;
        top: 1rem;
        right: 1rem;
        padding: 0.8rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .enabled {
        color: #27ae60;
    }

    .disabled {
        color: #e74c3c;
    }

    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            opacity: 1;
        }
    }

    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }

    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    .chat-container::-webkit-scrollbar-thumb {
        background: #bdc3c7;
        border-radius: 4px;
    }

    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #95a5a6;
    }

    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }

    .modal {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        width: 90%;
        max-width: 500px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.3s ease-out;
    }

    .modal h2 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }

    .modal-content {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .modal-actions {
        display: flex;
        justify-content: flex-end;
        gap: 1rem;
        margin-top: 1rem;
    }

    .cancel-btn {
        background: #95a5a6;
        color: white;
    }

    .cancel-btn:hover {
        background: #7f8c8d;
    }

    .upload-btn {
        background: #27ae60;
        color: white;
    }

    .upload-btn:disabled {
        background: #bdc3c7;
        cursor: not-allowed;
        transform: none;
    }

    .upload-btn:not(:disabled):hover {
        background: #219a52;
    }

    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .file-upload {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px dashed #bdc3c7;
        text-align: center;
        transition: all 0.3s ease;
    }

    .file-upload:hover {
        border-color: #3498db;
        background: #f0f7ff;
    }

    input[type="file"] {
        width: 100%;
    }

    /* Optional: Style the file input */
    input[type="file"]::file-selector-button {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: none;
        background: #3498db;
        color: white;
        cursor: pointer;
        margin-right: 1rem;
        transition: background 0.3s ease;
    }

    input[type="file"]::file-selector-button:hover {
        background: #2980b9;
    }

    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        text-align: center;
    }

    .loading-container h2 {
        margin: 1rem 0;
        color: #2c3e50;
    }

    .loading-container p {
        color: #7f8c8d;
        margin: 0;
    }

    .spinner {
        width: 64px;
        height: 64px;
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 8px 0;
    }

    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: #3498db;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }

    .typing-indicator span:nth-child(1) {
        animation-delay: -0.32s;
    }

    .typing-indicator span:nth-child(2) {
        animation-delay: -0.16s;
    }

    @keyframes bounce {
        0%,
        80%,
        100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-8px);
        }
    }

    .book-select {
        width: 100%;
        padding: 0.8rem;
        border: 2px solid #bdc3c7;
        border-radius: 8px;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        background: white;
        color: #2c3e50; /* Add this for text color */
    }
    .book-select option[value=""][disabled] {
        color: #7f8c8d;
    }

    .book-select:invalid {
        color: #7f8c8d;
    }

    .load-btn {
        width: 100%;
        background: #3498db;
        color: white;
        margin-top: 0.5rem;
    }
    .load-btn.loading {
        position: relative;
        cursor: not-allowed;
        opacity: 0.8;
    }

    .button-spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #ffffff;
        border-top: 2px solid transparent;
        border-radius: 50%;
        margin-right: 8px;
        animation: spin 1s linear infinite;
        vertical-align: middle;
    }

    .load-btn.loading .button-spinner {
        margin-right: 8px;
    }

    .load-btn {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    .divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 1rem 0;
    }

    .divider::before,
    .divider::after {
        content: "";
        flex: 1;
        border-bottom: 1px solid #bdc3c7;
    }

    .divider span {
        padding: 0 1rem;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
