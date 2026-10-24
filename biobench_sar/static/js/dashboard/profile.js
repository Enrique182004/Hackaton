import { $ } from "./api.js";

// Subject-specific wording. The template ships the math text (the server's default profile) so the
// first paint is right; biology swaps in once /api/config answers.
const WORDS = {
  math: {
    examples: ["I want to learn how to solve fractions", "Help me with the next step", "Quiz me on equations"],
    placeholder: "Ask about the math on your board, or what you want to learn…",
    filesTab: "Worksheets & notes",
    filesHint: "Add your worksheets and notes. Before answering, the assistant looks in the files you tick, and quiz problems come from them too. Untick a file to leave it out.",
    dropSub: "PDF · Word · CSV quiz banks · text · photos of worksheets",
    searchHint: "e.g. fractions",
    noFiles: "No worksheets or notes yet. The assistant answers from what it already knows and makes up quiz problems.",
    noneTicked: "No files ticked. The assistant won't look at your worksheets or notes.",
    filesBadge: "add worksheets",
    filesMissing: "No worksheets or notes yet",
    filesNoneTicked: "No worksheets or notes ticked",
    work: "your work",
  },
  biology: {
    examples: ["Is my flowchart missing a sterile step?", "Why does Leptosol runoff make the water cloudy?", "Where should I take my water samples?"],
    placeholder: "Ask about your drawing or your notes…",
    filesTab: "Study files",
    filesHint: "Add your notes, protocols or papers. Before answering, the assistant looks in the files you tick. Untick a file to leave it out.",
    dropSub: "PDF · Word · PowerPoint · CSV · text · photos of notes",
    searchHint: "e.g. autoclave",
    noFiles: "No files yet. The assistant answers from what it already knows.",
    noneTicked: "No files ticked. The assistant won't look at your notes.",
    filesBadge: "add your notes",
    filesMissing: "No study files yet",
    filesNoneTicked: "No study files ticked",
    work: "your drawing",
  },
};

// The math tutor modes as the student sees them (Ask badge, margin notes).
export const MODE_LABELS = { help: "step-by-step help", practice: "similar problem", quiz: "quiz" };

let profile = "math";
const listeners = [];

export function words() {
  return WORDS[profile];
}

export function currentProfile() {
  return profile;
}

export function onProfileChange(fn) {
  listeners.push(fn);
}

export function setProfile(name) {
  const previous = profile;
  profile = Object.hasOwn(WORDS, name) ? name : "math";
  const w = words();
  $("ask-input").placeholder = w.placeholder;
  $("files-tab-name").textContent = w.filesTab;
  $("files-hint").textContent = w.filesHint;
  $("drop-sub").textContent = w.dropSub;
  $("search-input").placeholder = w.searchHint;
  if (profile !== previous) for (const fn of listeners) fn(profile);
}
