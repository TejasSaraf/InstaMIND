import { useRef, useState } from "react";
import {
  ArrowRight,
  Bell,
  Brain,
  Camera,
  Mail,
  Shield,
  Video,
  WifiOff,
  Zap,
} from "lucide-react";

const GithubIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
  </svg>
);

const LinkedinIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
  </svg>
);

type LandingProps = {
  authError: string | null;
  googleButtonRef: (el: HTMLDivElement | null) => void;
};

const productSteps = [
  {
    icon: Camera,
    title: "Watch every feed",
    text: "Connect existing CCTV or IP cameras. The operator no longer has to stare at a wall of screens.",
  },
  {
    icon: Brain,
    title: "Detect incidents locally",
    text: "Fine-tuned Gemma 3 vision runs on-device to catch robbery, fighting, shooting, shoplifting, fainting, and normal scenes.",
  },
  {
    icon: Bell,
    title: "Escalate automatically",
    text: "InstaMIND routes the right response with incident type, confidence, evidence, and recommended action.",
  },
];

const differences = [
  {
    icon: Shield,
    title: "Private by default",
    text: "Frames are processed on the edge device. Surveillance video does not need to leave the building.",
  },
  {
    icon: WifiOff,
    title: "Works offline",
    text: "Local inference keeps the system useful when the network is unreliable or intentionally isolated.",
  },
  {
    icon: Zap,
    title: "Fast enough to matter",
    text: "The target is operational response time: frame capture to alert dispatch in under two seconds.",
  },
];

export default function Landing({ authError, googleButtonRef }: LandingProps) {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
  });
  const [formSubmitted, setFormSubmitted] = useState(false);

  const problemRef = useRef<HTMLElement>(null);
  const demoRef = useRef<HTMLElement>(null);
  const howRef = useRef<HTMLElement>(null);
  const contactRef = useRef<HTMLElement>(null);

  const scrollTo = (ref: React.RefObject<HTMLElement | null>) => {
    ref.current?.scrollIntoView({ behavior: "smooth" });
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setFormSubmitted(true);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-x-hidden">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        .font-heading { font-family: 'DM Sans', sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
      `}</style>

      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-[#0a0a0f]/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <button
            type="button"
            onClick={scrollToTop}
            className="font-mono text-lg font-bold tracking-tight text-white"
          >
            instaMIND
          </button>

          <div className="hidden md:flex items-center gap-8">
            <button
              onClick={() => scrollTo(problemRef)}
              className="text-sm text-neutral-400 hover:text-white transition"
            >
              Problem
            </button>
            <button
              onClick={() => scrollTo(demoRef)}
              className="text-sm text-neutral-400 hover:text-white transition"
            >
              Demo
            </button>
            <button
              onClick={() => scrollTo(howRef)}
              className="text-sm text-neutral-400 hover:text-white transition"
            >
              How it works
            </button>
            <button
              onClick={() => scrollTo(contactRef)}
              className="text-sm text-neutral-400 hover:text-white transition"
            >
              Contact
            </button>
          </div>

          <div className="relative flex items-center">
            <div
              ref={googleButtonRef}
              aria-label="Sign up"
              className="pointer-events-none absolute -left-[9999px] top-0 h-10 w-[110px] overflow-hidden opacity-0"
            />
            <a
              href="/instaMIND_Demo.mov"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded bg-amber-500 px-4 py-2.5 text-sm font-semibold text-black transition hover:bg-amber-400"
            >
              Watch Demo
            </a>
          </div>
        </div>
      </nav>

      <section className="pt-32 pb-16 px-6">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-[0.92fr_1.08fr] gap-12 items-center">
          <div className="text-left">
            <p className="font-mono text-xs uppercase tracking-[0.24em] text-amber-500 mb-5">
              AI security operator
            </p>
            <h1 className="font-heading text-4xl sm:text-5xl lg:text-6xl font-bold leading-[1.04] tracking-tight mb-6">
              Intelligence for security cameras.{" "}
              <span className="text-amber-500">
                AI security operators for every camera.
              </span>
            </h1>
            <p className="text-lg text-neutral-400 leading-relaxed max-w-xl mb-8">
              instaMIND watches CCTV feeds on-device, turning security footage
              into real-time incident alerts in under two seconds.
            </p>

            {authError && (
              <p className="mt-4 text-sm text-rose-400 bg-rose-500/10 border border-rose-500/20 px-4 py-2 rounded">
                {authError}
              </p>
            )}
          </div>

          <DemoVideo />
        </div>
      </section>

      <section className="border-y border-white/5 bg-[#0c0c12] py-5 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-sm text-neutral-400">
            Built for{" "}
            <span className="text-amber-500 font-semibold">
              Google DeepMind
            </span>
            <span className="text-white font-medium">
              {" "}
              × instaLILY AI Hackathon
            </span>
          </p>
        </div>
      </section>

      <section ref={problemRef} className="py-24 px-6">
        <div className="max-w-5xl mx-auto grid lg:grid-cols-[0.9fr_1.1fr] gap-12 items-start">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-amber-500 mb-4">
              Problem
            </p>
            <h2 className="font-heading text-3xl sm:text-4xl font-bold leading-tight">
              Security teams have more cameras than attention.
            </h2>
          </div>
          <div className="space-y-5 text-neutral-400 leading-relaxed">
            <p>
              A modern store, school, or public venue can have dozens of feeds.
              The operator is expected to notice a fight, robbery, medical
              collapse, or weapon in real time while nothing happens on most
              screens.
            </p>
            <p>
              That does not scale. Missed seconds become missed incidents. The
              camera already saw the event; the missing layer is an operator
              that watches every feed continuously and acts immediately.
            </p>
          </div>
        </div>
      </section>

      <section
        ref={howRef}
        className="py-24 px-6 bg-[#0c0c12] border-y border-white/5"
      >
        <div className="max-w-7xl mx-auto">
          <div className="max-w-2xl text-left mb-12">
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-amber-500 mb-4">
              Product
            </p>
            <h2 className="font-heading text-3xl sm:text-4xl font-bold mb-4">
              From camera feed to response, automatically.
            </h2>
            <p className="text-neutral-400">
              InstaMIND plugs into the surveillance stack you already have and
              adds local vision intelligence on top.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            {productSteps.map((step) => (
              <div
                key={step.title}
                className="bg-[#0a0a0f] border border-white/10 rounded-xl p-6 text-left"
              >
                <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-5">
                  <step.icon className="w-5 h-5 text-amber-500" />
                </div>
                <h3 className="font-heading text-xl font-semibold mb-3">
                  {step.title}
                </h3>
                <p className="text-sm text-neutral-400 leading-relaxed">
                  {step.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section ref={demoRef} className="py-24 px-6">
        <div className="max-w-5xl mx-auto text-left">
          <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-8">
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.22em] text-amber-500 mb-4">
                Demo
              </p>
              <h2 className="font-heading text-3xl sm:text-4xl font-bold">
                Actual dashboard. Local demo feeds.
              </h2>
            </div>
            <p className="text-sm text-neutral-500 max-w-sm">
              The product identifies people, tracks camera health, surfaces
              incidents, and routes alert actions from a single dashboard.
            </p>
          </div>
          <DemoVideo large />
        </div>
      </section>

      <section className="py-24 px-6 bg-[#0c0c12] border-y border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="max-w-2xl text-left mb-12">
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-amber-500 mb-4">
              Why now
            </p>
            <h2 className="font-heading text-3xl sm:text-4xl font-bold mb-4">
              Edge vision models are finally useful in security operations.
            </h2>
            <p className="text-neutral-400">
              The model is small enough to run locally, accurate enough to
              reduce operator load, and fast enough to change response time.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            {differences.map((item) => (
              <div
                key={item.title}
                className="border border-white/10 rounded-xl p-6 text-left bg-white/[0.02]"
              >
                <item.icon className="w-5 h-5 text-amber-500 mb-5" />
                <h3 className="font-heading text-lg font-semibold mb-2">
                  {item.title}
                </h3>
                <p className="text-sm text-neutral-400 leading-relaxed">
                  {item.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section ref={contactRef} className="py-24 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <Video className="w-6 h-6 text-amber-500 mx-auto mb-5" />
          <h2 className="font-heading text-3xl sm:text-4xl font-bold mb-4">
            Pilot InstaMIND on real camera feeds.
          </h2>
          <p className="text-neutral-400 mb-10">
            We are looking for pilot deployments in retail, schools, and public
            spaces where response time matters.
          </p>

          {formSubmitted ? (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-8">
              <div className="text-emerald-400 font-semibold text-lg mb-2">
                Request received.
              </div>
              <p className="text-neutral-400 text-sm">
                We will reach out within 24 hours to schedule your demo.
              </p>
            </div>
          ) : (
            <form onSubmit={handleFormSubmit} className="space-y-4 text-left">
              <div className="grid sm:grid-cols-2 gap-4">
                <input
                  type="text"
                  placeholder="Name"
                  required
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-amber-500/50 transition"
                />
                <input
                  type="email"
                  placeholder="Work email"
                  required
                  value={formData.email}
                  onChange={(e) =>
                    setFormData({ ...formData, email: e.target.value })
                  }
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-amber-500/50 transition"
                />
              </div>
              <input
                type="text"
                placeholder="Company"
                value={formData.company}
                onChange={(e) =>
                  setFormData({ ...formData, company: e.target.value })
                }
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-amber-500/50 transition"
              />
              <button
                type="submit"
                className="w-full bg-amber-500 hover:bg-amber-400 text-black font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
              >
                Request Pilot Access <ArrowRight className="w-4 h-4" />
              </button>
            </form>
          )}

          <div className="mt-12 pt-8 border-t border-white/5 text-left">
            <p className="text-sm text-neutral-500 mb-3">
              Built by{" "}
              <span className="text-white font-medium">Tejas Saraf</span>
            </p>
            <div className="flex items-center gap-4">
              <a
                href="mailto:tejassaraf25@gmail.com"
                className="text-neutral-500 hover:text-amber-500 transition"
                aria-label="Email"
              >
                <Mail className="w-4 h-4" />
              </a>
              <a
                href="https://github.com/tejassaraf"
                target="_blank"
                rel="noopener noreferrer"
                className="text-neutral-500 hover:text-amber-500 transition"
                aria-label="GitHub"
              >
                <GithubIcon className="w-4 h-4" />
              </a>
              <a
                href="https://linkedin.com/in/tejassaraf"
                target="_blank"
                rel="noopener noreferrer"
                className="text-neutral-500 hover:text-amber-500 transition"
                aria-label="LinkedIn"
              >
                <LinkedinIcon className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-white/5 py-6 px-6">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-mono text-sm font-bold text-white">
              instaMIND
            </span>
            <span className="text-xs text-neutral-600">
              © 2025 instaMIND. All rights reserved.
            </span>
          </div>
          <a
            href="https://github.com/tejassaraf/instaMIND"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-neutral-500 hover:text-white transition flex items-center gap-1.5"
          >
            <GithubIcon className="w-3.5 h-3.5" /> GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}

function DemoVideo({ large = false }: { large?: boolean }) {
  return (
    <div className="relative">
      <div className="absolute -inset-4 bg-gradient-to-r from-amber-500/10 to-transparent rounded-2xl blur-3xl" />
      <div
        className={`relative bg-[#0d0d14] border border-white/10 rounded-xl p-3 ${
          large ? "shadow-2xl" : ""
        }`}
      >
        <video
          className="w-full aspect-video rounded-lg bg-black"
          src="/LandingPage_Demo.mp4"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
        />
      </div>
    </div>
  );
}
