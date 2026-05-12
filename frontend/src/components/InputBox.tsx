import { useRef } from "react";

type InputBoxProps = {
  onFileSelect: (file: File | null) => void;
  drag: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  selectedFileName: string | null;
};

function IconUpload({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
      />
    </svg>
  );
}

function IconCheck({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  );
}

export function InputBox({
  onFileSelect,
  drag,
  onDragOver,
  onDragLeave,
  onDrop,
  selectedFileName,
}: InputBoxProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    onFileSelect(file ?? null);
    e.target.value = "";
  };

  return (
    <div className="rounded-xl overflow-hidden">
      <div className="p-2">
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`rounded-xl border-1 border-dashed text-center transition-colors cursor-pointer ${
            selectedFileName ? "p-2" : "p-12"
          } ${
            drag
              ? "border-purple-500 bg-purple-500/5"
              : selectedFileName
              ? "border-gray-400 bg-gray-950/20"
              : "border-neutral-700 hover:border-neutral-600 bg-neutral-900/50"
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
            aria-label="Upload video"
          />

          {selectedFileName ? (
            <div className="flex flex-col items-center gap-2">
              <IconCheck className="w-2 h-2 text-gray-400" />
              <p className="text-sm font-medium text-gray-300 truncate max-w-xs">
                {selectedFileName}
              </p>
              <p className="text-xs text-neutral-500">
                Press play below to start analysis
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <IconUpload className="w-4 h-4 text-neutral-400" />
              <p className="text-sm font-medium text-neutral-300">
                Click or drag a video to begin
              </p>
              <p className="text-xs text-neutral-500">
                MP4, MOV, AVI, MKV, WebM — press play to start live analysis
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
