import { setContext, t } from "../i18n.js";

// The subject decides the wording; i18n.js holds it for each language. The template ships the math text (the
// server's default profile) so the first paint is right; biology swaps in once /api/config answers.
const PROFILES = ["math", "biology"];

let profile = "math";
// False until the server has said which subject it runs; the math mode picker waits for it.
let known = false;
const listeners = [];

// Read at render time, so the words follow both the subject and the language.
export function words() {
  return {
    examples: [1, 2, 3].map((n) => t(`examples.${profile}.${n}`)),
    noFiles: t(`files.noFiles.${profile}`),
    noneTicked: t(`files.noneTicked.${profile}`),
    filesBadge: t(`files.badge.add.${profile}`),
    filesMissing: t(`check.files.missing.${profile}`),
    filesNoneTicked: t(`check.files.noneTicked.${profile}`),
    work: t(`work.${profile}`),
  };
}

// The math tutor modes as the student sees them (Ask badge, margin notes).
export function modeLabel(mode) {
  return t(`mode.${mode}`);
}

export function currentProfile() {
  return profile;
}

export function profileKnown() {
  return known;
}

export function onProfileChange(fn) {
  listeners.push(fn);
}

// Safe to call on every status poll: listeners only run when the subject is first known or changes.
export function setProfile(name) {
  const previous = profile;
  const first = !known;
  known = true;
  profile = PROFILES.includes(name) ? name : "math";
  if (!first && profile === previous) return;
  // The template's data-i18n="….{profile}" texts (tab name, hints, placeholders) follow the subject.
  setContext({ profile });
  for (const fn of listeners) fn(profile);
}
