type NavView = "dashboard" | "upload" | "realtime";

type NavbarProps = {
  activeView: NavView;
  onChangeView: (view: NavView) => void;
  authInitials: string | null;
  onSignOut: () => void;
};

export function Navbar({
  activeView,
  onChangeView,
  authInitials,
  onSignOut,
}: NavbarProps) {
  return (
    <nav className="sticky top-0 z-20 border-b border-neutral-800 bg-black/95 backdrop-blur-md shadow-lg shadow-black/20">
      <div className="px-10">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center h-14 sm:h-16 gap-4">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-600 font-bold tracking-tight text-white">
              instaMIND
            </span>
          </div>
          <div className="inline-flex items-center gap-1 sm:gap-2">
            <button
              type="button"
              onClick={() => onChangeView("dashboard")}
              className={`inline-flex items-center gap-1.5 px-3 sm:px-4 py-1.5 text-xs sm:text-sm rounded-md transition ${
                activeView === "dashboard"
                  ? "text-white"
                  : "text-neutral-300 hover:text-white"
              }`}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M3 10.5L12 3l9 7.5V20a1 1 0 0 1-1 1h-5.5v-6h-5v6H4a1 1 0 0 1-1-1v-9.5Z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Dashboard
            </button>
            <button
              type="button"
              onClick={() => onChangeView("upload")}
              className={`inline-flex items-center gap-1.5 px-3 sm:px-4 py-1.5 text-xs sm:text-sm rounded-md transition ${
                activeView === "upload"
                  ? "text-white"
                  : "text-neutral-300 hover:text-white"
              }`}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M12 16V5M12 5l-4 4M12 5l4 4M4 16.5V19a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2.5"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Upload
            </button>
            <button
              type="button"
              onClick={() => onChangeView("realtime")}
              className={`inline-flex items-center gap-1.5 px-3 sm:px-4 py-1.5 text-xs sm:text-sm rounded-md transition ${
                activeView === "realtime"
                  ? "text-white"
                  : "text-neutral-300 hover:text-white"
              }`}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                aria-hidden="true"
              >
                <rect
                  x="4"
                  y="4"
                  width="16"
                  height="16"
                  rx="2.5"
                  stroke="currentColor"
                  strokeWidth="1.8"
                />
                <path
                  d="M10 8.8v6.4l5-3.2-5-3.2Z"
                  fill="currentColor"
                  stroke="currentColor"
                  strokeWidth="0.6"
                  strokeLinejoin="round"
                />
              </svg>
              RealTime
            </button>
          </div>
          <div className="justify-self-end min-w-[220px] flex flex-col items-end gap-1">
            <div className="inline-flex items-center gap-2">
              <span
                className="w-8 h-8 rounded-full inline-flex items-center justify-center text-[11px] font-semibold text-white border"
                style={{ backgroundColor: "#6b7280", borderColor: "#6b7280" }}
              >
                {authInitials ?? "U"}
              </span>
              <button
                type="button"
                onClick={onSignOut}
                className="text-xs text-neutral-300 hover:text-white border border-neutral-700 rounded px-2 py-1 transition"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
