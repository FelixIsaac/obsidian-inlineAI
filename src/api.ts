// api.ts
import { ChatOpenAI, AzureChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
	SystemMessage,
	HumanMessage,
	AIMessage,
} from "@langchain/core/messages";
import { InlineAISettings } from "./settings";
import { App, MarkdownView, Notice } from "obsidian";
import { EditorView } from "@codemirror/view";
import { callCodexApi } from "./codex-client";
import { getValidCodexToken, CodexTokens } from "./codex-auth";
import { setGeneratedResponseEffect } from "./modules/AIExtension";
import { parseCommand } from "./modules/commands/parser";
import { MessageQueue } from "./modules/messageHistory/queue";

const MESSAGE_HISTORY_LIMIT = 20;

export type HistoryMessage = {
	mode: string;
	userPrompt: string;
};

/**
 * Class to manage interactions with different chat APIs.
 */
export class ChatApiManager {
	private chatClient:
		| ChatOpenAI
		| ChatOllama
		| ChatGoogleGenerativeAI
		| AzureChatOpenAI
		| null;
	private app: App;
	private settings: InlineAISettings;
	private messageHistory: MessageQueue<HistoryMessage>;
	/**
	 * Initializes the ChatApiManager with the given settings.
	 * @param settings - Configuration settings for the chat API.
	 * @param app - The Obsidian App instance.
	 */
	constructor(settings: InlineAISettings, app: App) {
		this.app = app;
		this.chatClient = this.initializeChatClient(settings);
		this.settings = settings;
		this.messageHistory = new MessageQueue<HistoryMessage>(
			MESSAGE_HISTORY_LIMIT,
		);
	}

	/**
	 * Extracts the instance name from an Azure endpoint URL.
	 * @param endpoint - The Azure endpoint URL.
	 * @returns The instance name or null if invalid.
	 */
	private extractAzureInstanceName(endpoint: string): string | null {
		const trimmedEndpoint = endpoint.trim();

		// Match both openai.azure.com and cognitiveservices.azure.com formats
		const openaiMatch = trimmedEndpoint.match(
			/https:\/\/([^.]+)\.openai\.azure\.com/,
		);
		if (openaiMatch) {
			return openaiMatch[1];
		}

		const cognitiveservicesMatch = trimmedEndpoint.match(
			/https:\/\/([^.]+)\.cognitiveservices\.azure\.com/,
		);
		if (cognitiveservicesMatch) {
			return cognitiveservicesMatch[1];
		}

		return null;
	}

	/**
	 * Initializes the appropriate chat client based on the provider specified in settings.
	 * @param settings - Configuration settings for the chat API.
	 * @returns An instance of ChatOpenAI, ChatOllama, AzureChatOpenAI, or null if initialization fails.
	 */
	private initializeChatClient(
		settings: InlineAISettings,
	):
		| ChatOpenAI
		| ChatOllama
		| ChatGoogleGenerativeAI
		| AzureChatOpenAI
		| null {
		try {
			if (settings.messageHistory) {
				this.messageHistory = new MessageQueue<HistoryMessage>(MESSAGE_HISTORY_LIMIT);
				try {
					const saved = localStorage.getItem("inlineai-prompt-history");
					if (saved) {
						const items: HistoryMessage[] = JSON.parse(saved);
						items.forEach((item) => this.messageHistory.enqueue(item));
					}
				} catch {}
			} else {
				this.messageHistory = new MessageQueue<HistoryMessage>(0);
			}

			switch (settings.provider) {
				case "openai":
					if (!settings.apiKey) {
						new Notice(
							"⚠️ OpenAI API key is required. Please check your settings.",
						);
						return null;
					}
					return new ChatOpenAI({
						modelName: settings.model,
						temperature: 0,
						apiKey: settings.apiKey,
					});

				case "ollama":
					return new ChatOllama({
						model: settings.model,
					});
				case "gemini":
					return new ChatGoogleGenerativeAI({
						model: settings.model,
						apiKey: settings.apiKey,
					});
				case "azure":
					if (!settings.apiKey || !settings.azureEndpoint) {
						new Notice(
							"⚠️ API key and Azure endpoint are required for Azure provider.",
						);
						return null;
					}

					// Extract instance name from the endpoint URL
					const instanceName = this.extractAzureInstanceName(
						settings.azureEndpoint,
					);
					if (!instanceName) {
						new Notice(
							"⚠️ Invalid Azure endpoint format. Expected: https://your-resource.openai.azure.com",
						);
						return null;
					}

					return new AzureChatOpenAI({
						azureOpenAIApiKey: settings.apiKey,
						azureOpenAIApiInstanceName: instanceName,
						azureOpenAIApiDeploymentName: settings.model,
						azureOpenAIApiVersion:
							settings.azureApiVersion || "2024-02-15-preview",
						temperature: 0,
					});
				case "custom":
					if (!settings.apiKey || !settings.customURL) {
						new Notice(
							"⚠️ API key and custom base URL are required for custom providers.",
						);
						return null;
					}
					return new ChatOpenAI({
						modelName: settings.model,
						temperature: 0,
						openAIApiKey: settings.apiKey,
						// 'configuration.basePath' is the recognized property
						configuration: {
							baseURL: settings.customURL.trim(),
						},
					});

				case "codex":
					// Handled directly in callApi — no LangChain client needed
					return null;

				default:
					new Notice(`⚠️ Unsupported provider: ${settings.provider}`);
					return null;
			}
		} catch (error: any) {
			console.error("Error initializing chat client:", error);
			new Notice(`❌ Error initializing chat client: ${error.message}`);
			return null;
		}
	}

	/**
	 * Calls the chat API with the provided content and context.
	 * @param systemMessage - The system message to send to the chat API.
	 * @param message - The user's message to send to the chat API.
	 * @returns A promise that resolves with the generated content or an error message.
	 */
	public async callApi(
		systemMessage: string,
		message: string,
	): Promise<string> {
		if (this.settings.provider === "codex") {
			return this.callCodexProvider(systemMessage, message);
		}

		if (!this.chatClient) {
			new Notice(
				"⚠️ Chat client is not initialized. Please check your settings.",
			);
			return "⚠️ Chat client is not available.";
		}

		const messages = [
			new SystemMessage(systemMessage),
			new HumanMessage(message),
		];

		try {
			const aiMessage = await this.chatClient.invoke(messages);
			if (typeof aiMessage === "string") {
				return aiMessage;
			}
			return aiMessage.content.toString();
		} catch (error: any) {
			console.error("Error calling the chat model:", error);
			new Notice(`❌ Error calling the chat model: ${error.message}`);
			return "⚠️ Failed to generate a response. Please try again later.";
		}
	}

	private async callCodexProvider(
		systemMessage: string,
		message: string,
	): Promise<string> {
		const s = this.settings;
		if (!s.codexAccess || !s.codexRefresh || !s.codexAccountId) {
			new Notice(
				"⚠️ Codex: not signed in — open Settings → InlineAI and click 'Sign in with ChatGPT'",
			);
			return "⚠️ Codex not authenticated.";
		}

		try {
			const tokens: CodexTokens = {
				access: s.codexAccess,
				refresh: s.codexRefresh,
				expires: s.codexExpires ?? 0,
				accountId: s.codexAccountId,
			};

			const accessToken = await getValidCodexToken(
				tokens,
				async (refreshed) => {
					this.settings.codexAccess = refreshed.access;
					this.settings.codexRefresh = refreshed.refresh;
					this.settings.codexExpires = refreshed.expires;
				},
			);

			if (!accessToken) {
				new Notice("⚠️ Codex: session expired — please sign in again");
				return "⚠️ Codex session expired.";
			}

			return await callCodexApi(
				systemMessage,
				message,
				accessToken,
				s.codexAccountId,
				s.model,
			);
		} catch (error: any) {
			console.error("Codex error:", error);
			new Notice(`❌ Codex: ${error.message}`);
			return "⚠️ Codex request failed.";
		}
	}

	/**
	 * Handles user input and updates the editor with the response.
	 * @param systemPrompt - The system prompt to send to the chat API.
	 * @param userRequest - The user's request to process.
	 * @returns The AI-generated response or an error message.
	 */
	private async handleEditorUpdate(
		systemPrompt: string,
		userRequest: string,
	): Promise<string> {
		try {
			const response = await this.callApi(systemPrompt, userRequest);
			if (!response) return "⚠️ No response generated.";

			const markdownView =
				this.app.workspace.getActiveViewOfType(MarkdownView);
			if (!markdownView) {
				new Notice("⚠️ No active Markdown editor found.");
				return "";
			}

			const mainEditorView = (markdownView.editor as any)
				.cm as EditorView;
			mainEditorView?.dispatch({
				effects: setGeneratedResponseEffect.of({
					airesponse: response,
					prompt: userRequest,
				}),
			});

			return response;
		} catch (error: any) {
			console.error("Error processing request:", error);
			new Notice(`❌ Error processing request: ${error.message}`);
			return "⚠️ Failed to process request.";
		}
	}
	private detectDocumentType(filename: string, doc: string): string {
		if (/\d{4}-\d{2}-\d{2}/.test(filename)) return "daily-note";
		if (/meeting|minutes|standup|1-on-1|1on1/i.test(filename + doc.slice(0, 500))) return "meeting-note";
		const codeBlocks = (doc.match(/```/g) ?? []).length;
		if (codeBlocks >= 6) return "code-note";
		if (/book|literature|reading|summary|review/i.test(filename)) return "literature-note";
		return "";
	}

	private extractNoteContext(selectionText: string): string {
		try {
			const file = this.app.workspace.getActiveFile();
			const noteTitle = file?.basename ?? "";
			const markdownView =
				this.app.workspace.getActiveViewOfType(MarkdownView);
			if (!markdownView) return "";

			const cm = (markdownView.editor as any).cm as EditorView;
			const doc = cm.state.doc.toString();
			const cursor = cm.state.selection.main.from;

			// Extract frontmatter metadata
			let frontmatterContext = "";
			if (doc.startsWith("---")) {
				const fmEnd = doc.indexOf("\n---", 3);
				if (fmEnd !== -1) {
					const fm = doc.slice(3, fmEnd).trim();
					const relevantLines = fm.split("\n").filter((line) =>
						/^(tags|type|status|aliases|category|topic):/i.test(line.trim()),
					);
					if (relevantLines.length > 0) {
						frontmatterContext = `Frontmatter: ${relevantLines.join(", ")}`;
					}
				}
			}

			// Find nearest heading above cursor
			const docBeforeCursor = doc.slice(0, cursor);
			const lines = docBeforeCursor.split("\n");
			let nearestHeading = "";
			for (let i = lines.length - 1; i >= 0; i--) {
				if (/^#{1,3}\s/.test(lines[i])) {
					nearestHeading = lines[i].replace(/^#+\s*/, "").trim();
					break;
				}
			}

			// Get surrounding paragraphs (split by blank lines)
			const selectionStart = doc.indexOf(
				selectionText,
				Math.max(0, cursor - selectionText.length - 200),
			);
			const before =
				selectionStart > 0
					? doc.slice(0, selectionStart)
					: docBeforeCursor;
			const after =
				selectionStart >= 0
					? doc.slice(selectionStart + selectionText.length)
					: doc.slice(cursor);

			const beforeParas = before
				.split(/\n\n+/)
				.filter((p) => p.trim())
				.slice(-3);
			const afterParas = after
				.split(/\n\n+/)
				.filter((p) => p.trim())
				.slice(0, 3);

			if (beforeParas.length === 0 && afterParas.length === 0) return "";

			const MAX_CONTEXT_CHARS = 1500;
			let contextStr = "";
			if (noteTitle) contextStr += `Note: ${noteTitle}\n`;
			if (nearestHeading) contextStr += `Section: ${nearestHeading}\n`;
			if (frontmatterContext) contextStr += `${frontmatterContext}\n`;
			const docType = this.detectDocumentType(noteTitle, doc);
			const typeHints: Record<string, string> = {
				"daily-note": "Document type: daily note — prefer concise bullets and task items.",
				"meeting-note": "Document type: meeting note — prefer action items, decisions, attendees.",
				"code-note": "Document type: technical/code note — prefer technical precision and code blocks.",
				"literature-note": "Document type: literature note — prefer quotation-aware, citation-friendly responses.",
			};
			if (typeHints[docType]) contextStr += `${typeHints[docType]}\n`;
			if (beforeParas.length > 0)
				contextStr += `\nContext before:\n${beforeParas.join("\n\n")}`;
			if (afterParas.length > 0)
				contextStr += `\n\nContext after:\n${afterParas.join("\n\n")}`;

			if (contextStr.length > MAX_CONTEXT_CHARS) {
				contextStr = contextStr.slice(0, MAX_CONTEXT_CHARS) + "\n[…]";
			}

			return contextStr.trim();
		} catch {
			return "";
		}
	}

	/**
	 * Processes selected text using the specified prompt and transformation.
	 * @param userPrompt - The transformation prompt (e.g., "Add Emojis").
	 * @param selectedText - The selected text to transform.
	 * @returns The transformed text or an error message.
	 */
	public async callSelection(
		userPrompt: string,
		selectedText: string,
	): Promise<string> {
		userPrompt = parseCommand(
			userPrompt,
			this.settings.commandPrefix,
			this.settings.customCommands,
		);

		let isCursor = false;
		if (selectedText.trim().length === 0) {
			isCursor = true;
		}

		const systemPrompt = isCursor
			? this.settings.cursorPrompt
			: this.settings.selectionPrompt;
		let finalUserPrompt = ``;
		const mode = isCursor ? "cursor" : "selection";
		if (this.settings.messageHistory) {
			this.messageHistory.enqueue({ mode, userPrompt });
			try {
				const items = this.messageHistory.getItems();
				localStorage.setItem("inlineai-prompt-history", JSON.stringify(items.slice(-20)));
			} catch {}
		}

		if (isCursor) {
			finalUserPrompt = `
      **Task:** ${userPrompt}  
      **Output:**`;
		} else {
			finalUserPrompt = `
      **Task:** ${userPrompt}  
      **Input:**  
      ${selectedText}

      **Output:**`;
		}
		const noteContext = this.extractNoteContext(selectedText);
		const enhancedSystemPrompt = noteContext
			? `${systemPrompt}\n\n---\nDocument context (for reference only — do not include in output):\n${noteContext}`
			: systemPrompt;

		const selLen = selectedText.trim().length;
		let scopeHint = "";
		if (selLen > 0 && selLen < 80) {
			scopeHint = "\n\nScope: The selection is a single word or short phrase. Output must be equally brief — match the selection length exactly.";
		} else if (selLen < 400) {
			scopeHint = "\n\nScope: The selection is a sentence or two. Output should be roughly the same length — one to two sentences.";
		} else if (selLen < 1500) {
			scopeHint = "\n\nScope: The selection is a paragraph. Output should be roughly one paragraph.";
		} else if (selLen >= 1500) {
			scopeHint = "\n\nScope: The selection is multiple paragraphs. Match the structure and length of the input.";
		}
		const finalSystemPrompt = enhancedSystemPrompt + scopeHint;

		return this.handleEditorUpdate(finalSystemPrompt, finalUserPrompt);
	}

	/**
	 * Updates the manager's settings and reinitializes the chat client.
	 * @param settings - New configuration settings for the chat API.
	 */
	public updateSettings(settings: InlineAISettings): void {
		this.settings = settings;
		const newChatClient = this.initializeChatClient(settings);
		if (!newChatClient) {
			return;
		}
		this.chatClient = newChatClient;
	}

	public getMessageHistory(): HistoryMessage[] {
		return this.messageHistory.getItems();
	}
}
