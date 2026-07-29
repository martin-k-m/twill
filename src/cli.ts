#!/usr/bin/env node
// cli.ts — the `aster` command: run a script or start the REPL.

import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { Interpreter, AsterRuntimeError } from "./interpreter.ts";
import { AsterSyntaxError } from "./lexer.ts";
import { formatValue, UNIT } from "./values.ts";

const VERSION = "0.1.0";

function main(): void {
  const args = process.argv.slice(2);

  if (args.includes("--version") || args.includes("-v")) {
    console.log(`Aster ${VERSION}`);
    return;
  }
  if (args.includes("--help") || args.includes("-h")) {
    printHelp();
    return;
  }

  const file = args.find((a) => !a.startsWith("-"));
  if (file) {
    runFile(file);
  } else {
    repl();
  }
}

function runFile(path: string): void {
  let source: string;
  try {
    source = readFileSync(path, "utf8");
  } catch {
    console.error(`aster: cannot read file '${path}'`);
    process.exitCode = 1;
    return;
  }
  const interp = new Interpreter();
  try {
    interp.run(source);
  } catch (e) {
    reportError(e);
    process.exitCode = 1;
  }
}

function repl(): void {
  console.log(`Aster ${VERSION} — a tensor-first, differentiable language.`);
  console.log("Type an expression, or :help / :quit.");
  const interp = new Interpreter();
  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "aster> " });
  rl.prompt();

  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (trimmed === ":quit" || trimmed === ":q") {
      rl.close();
      return;
    }
    if (trimmed === ":help") {
      printHelp();
      rl.prompt();
      return;
    }
    if (trimmed.length > 0) {
      try {
        const value = interp.run(trimmed);
        if (value !== UNIT) console.log(formatValue(value));
      } catch (e) {
        reportError(e);
      }
    }
    rl.prompt();
  });

  rl.on("close", () => process.exit(0));
}

function reportError(e: unknown): void {
  if (e instanceof AsterSyntaxError || e instanceof AsterRuntimeError) {
    console.error(`${e.name}: ${e.message}`);
  } else if (e instanceof Error) {
    console.error(`error: ${e.message}`);
  } else {
    console.error(`error: ${String(e)}`);
  }
}

function printHelp(): void {
  console.log(`Aster ${VERSION}

Usage:
  aster <file.ast>     Run an Aster program
  aster                Start the interactive REPL
  aster --version      Print the version
  aster --help         Show this help

In the REPL: :help, :quit`);
}

main();
