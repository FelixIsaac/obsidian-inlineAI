import { SlashCommand, BUILT_IN_COMMANDS } from "./source";

export function parseCommand(
	userInput: string,
	prefix: string,
	customCommands: SlashCommand[],
): string {
	const allCommands = [...BUILT_IN_COMMANDS, ...customCommands];
	let result = userInput;
	for (const command of allCommands) {
		const commandPattern = `${prefix}${command.keyword}`;
		if (result.includes(commandPattern)) {
			result = result.replace(commandPattern, command.prompt);
		}
	}
	return result;
}
