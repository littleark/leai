import { MediaRecorder, register } from "extendable-media-recorder";
import { connect } from "extendable-media-recorder-wav-encoder";

export class AudioTranscriptionClient {
  static isEncoderRegistered = false;

  constructor(serverUrl) {
    this.serverUrl = serverUrl;
    this.socket = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.chunkCounter = 0;  // Add counter for debugging
  }

  async startTranscription() {
    console.log("Starting transcription process...");
    await this.cleanup();

    try {
      // Register encoder only if not already registered
      if (!AudioTranscriptionClient.isEncoderRegistered) {
        console.log("Registering WAV encoder...");
        await register(await connect());
        AudioTranscriptionClient.isEncoderRegistered = true;
        console.log("WAV encoder registered successfully");
      }

      console.log("Creating WebSocket connection to:", this.serverUrl);
      this.socket = new WebSocket(this.serverUrl);

      this.socket.onopen = () => {
        console.log("WebSocket connection established, starting recording...");
        this.startRecording();
      };

      this.socket.onmessage = (event) => {
        console.log("Received message from server:", event.data);
        const transcriptionResult = JSON.parse(event.data);

        if (transcriptionResult.type === "final_transcript") {
          console.log("Final Transcript:", transcriptionResult.transcript);
        } else {
          this.processTranscription(transcriptionResult);
        }
      };

      this.socket.onerror = (error) => {
        console.error("WebSocket Error:", error);
        this.cleanup();
      };

      this.socket.onclose = (event) => {
        console.log("WebSocket closed with code:", event.code, "reason:", event.reason);
        this.cleanup();
      };

    } catch (error) {
      console.error("Error in startTranscription:", error);
    }
  }

  async startRecording() {
    try {
      console.log("Requesting microphone access...");
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 2,
          sampleRate: 44100,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });
      console.log("Microphone access granted");

      const audioContext = new AudioContext({ sampleRate: 44100 });
      console.log("Audio context created with sample rate:", audioContext.sampleRate);

      const mediaStreamAudioSourceNode = new MediaStreamAudioSourceNode(
        audioContext,
        { mediaStream: this.stream }
      );
      const mediaStreamAudioDestinationNode = new MediaStreamAudioDestinationNode(audioContext);

      mediaStreamAudioSourceNode.connect(mediaStreamAudioDestinationNode);
      console.log("Audio nodes connected");

      this.mediaRecorder = new MediaRecorder(
        mediaStreamAudioDestinationNode.stream,
        {
          mimeType: "audio/wav",
          sampleRate: 44100,
        }
      );
      console.log("MediaRecorder created with settings:", this.mediaRecorder.settings);

      this.mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.chunkCounter++;
          console.log(`Sending audio chunk #${this.chunkCounter}, size: ${event.data.size} bytes`);

          // Debug: Log the first few chunks' content
          if (this.chunkCounter <= 3) {
            const arrayBuffer = await event.data.arrayBuffer();
            console.log(`Chunk #${this.chunkCounter} first few bytes:`,
              new Uint8Array(arrayBuffer).slice(0, 10));
          }

          this.socket.send(event.data);
        }
      };

      this.mediaRecorder.onstart = () => {
        console.log("MediaRecorder started");
      };

      this.mediaRecorder.onstop = () => {
        console.log("MediaRecorder stopped");
      };

      this.mediaRecorder.onerror = (error) => {
        console.error("MediaRecorder error:", error);
      };

      console.log("Starting MediaRecorder with 100ms timeslice");
      this.mediaRecorder.start(100);
    } catch (error) {
      console.error("Error in startRecording:", error);
      console.error("Full error object:", JSON.stringify(error, Object.getOwnPropertyNames(error)));
      this.cleanup();
    }
  }

  async cleanup() {
    console.log("Starting cleanup...");

    if (this.mediaRecorder) {
      console.log("MediaRecorder state before cleanup:", this.mediaRecorder.state);
      if (this.mediaRecorder.state !== 'inactive') {
        try {
          this.mediaRecorder.stop();
          console.log("MediaRecorder stopped");
        } catch (e) {
          console.log('MediaRecorder already stopped:', e);
        }
      }
      this.mediaRecorder = null;
    }

    if (this.stream) {
      const tracks = this.stream.getTracks();
      console.log(`Stopping ${tracks.length} audio tracks...`);
      tracks.forEach((track, index) => {
        track.stop();
        console.log(`Track ${index} stopped`);
      });
      this.stream = null;
    }

    if (this.socket) {
      console.log("WebSocket state before cleanup:", this.socket.readyState);
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.close();
        console.log("WebSocket closed");
      }
      this.socket = null;
    }

    this.chunkCounter = 0;
    console.log("Cleanup completed");
  }

  async stopTranscription() {
    console.log("Stopping transcription...");
    await this.cleanup();
  }

  processTranscription(result) {
    console.log("Processing transcription result:", result);
    if (result.is_final) {
      console.log(
        "Final Transcription:",
        result.channel.alternatives[0].transcript
      );
    } else {
      console.log(
        "Interim Transcription:",
        result.channel.alternatives[0].transcript
      );
    }
  }
}
