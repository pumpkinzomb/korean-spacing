declare module 'node:path' {
  export function join(...segments: string[]): string;
  export function resolve(...segments: string[]): string;
}

declare module 'node:fs' {
  export function existsSync(path: string): boolean;
}

declare const __dirname: string;

