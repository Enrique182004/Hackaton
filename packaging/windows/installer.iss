; Inno Setup 6 script: per-user installer for the Windows build (no administrator rights needed).
;
;   pyinstaller packaging\sar.spec --noconfirm
;   iscc /DAppVersion=0.1.0 packaging\windows\installer.iss
;
; Reads   dist\SAR Whiteboard Assistant\   (the PyInstaller one-folder build)
; Writes  dist\SAR-Whiteboard-Assistant-Setup-<version>.exe
; Installs to %LOCALAPPDATA%\Programs\SAR Whiteboard Assistant; the app keeps its data in
; %LOCALAPPDATA%\SAR Whiteboard Assistant, which the uninstaller leaves alone (calibration, uploads).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef DistDir
  #define DistDir "..\..\dist"
#endif
#define AppName "SAR Whiteboard Assistant"
#define AppExe "SAR Whiteboard Assistant.exe"

[Setup]
AppId={{4B46C5CA-650C-4889-8529-A9D1A7BEB35E}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=SAR Whiteboard Assistant contributors
AppPublisherURL=https://github.com/Enrique182004/Hackaton
AppSupportURL=https://github.com/Enrique182004/Hackaton/blob/main/docs/DESKTOP_BUILDS.md
VersionInfoVersion={#AppVersion}
; Per-user: {autopf} becomes %LOCALAPPDATA%\Programs and shortcuts go to the user's Start menu.
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#DistDir}
OutputBaseFilename=SAR-Whiteboard-Assistant-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
WizardStyle=modern
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Dirs]
Name: "{localappdata}\{#AppName}"

[Files]
Source: "{#DistDir}\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Template for room settings: copy it to sar.env (next to the exe, or in the data folder) and edit.
Source: "..\..\sar.env.example"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{autoprograms}\{#AppName} data folder"; Filename: "{localappdata}\{#AppName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
