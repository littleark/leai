import { MediaRecorder, register } from "extendable-media-recorder";
import { connect } from "extendable-media-recorder-wav-encoder";

export class AudioTranscriptionClient {
  static isEncoderRegistered = false;

  constructor(serverUrl) {
    this.serverUrl = serverUrl;
    this.socket = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;  // Add this to track the media stream
  }

  async startTranscription() {
    // Clean up any existing connections first
    await this.cleanup();

    // Register encoder only if not already registered
    if (!AudioTranscriptionClient.isEncoderRegistered) {
      await register(await connect());
      AudioTranscriptionClient.isEncoderRegistered = true;
    }

    this.socket = new WebSocket(this.serverUrl);

    this.socket.onopen = () => {
      console.log("WebSocket connection established");
      this.startRecording();
    };

    this.socket.onmessage = (event) => {
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

    this.socket.onclose = () => {
      console.log("WebSocket connection closed");
      this.cleanup();
    };
  }

  async startRecording() {
    try {
      // this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 2,
          sampleRate: 44100,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      const audioContext = new AudioContext({ sampleRate: 44100 });
      const mediaStreamAudioSourceNode = new MediaStreamAudioSourceNode(
        audioContext,
        { mediaStream: this.stream }
      );
      const mediaStreamAudioDestinationNode = new MediaStreamAudioDestinationNode(audioContext);

      mediaStreamAudioSourceNode.connect(mediaStreamAudioDestinationNode);

      this.mediaRecorder = new MediaRecorder(
        mediaStreamAudioDestinationNode.stream,
        {
          mimeType: "audio/wav",
          sampleRate: 44100,
        }
      );

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(event.data);
        }
      };

      this.mediaRecorder.start(100);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      this.cleanup();
    }
  }

  async cleanup() {
    // Stop media recorder
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        this.mediaRecorder.stop();
      } catch (e) {
        console.log('MediaRecorder already stopped');
      }
      this.mediaRecorder = null;
    }

    // Stop all tracks in the media stream
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    // Close WebSocket connection
    if (this.socket) {
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.close();
      }
      this.socket = null;
    }
  }

  async stopTranscription() {
    await this.cleanup();
  }

  processTranscription(result) {
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

// Usage example
// const transcriptionClient = new AudioTranscriptionClient("ws://localhost:8765");
// transcriptionClient.startTranscription();
