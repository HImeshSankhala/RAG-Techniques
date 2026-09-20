/**
 * The shared formatters. Cheap to cover, and the reason they were extracted was
 * that three copies had started to drift — these tests are what stops the next
 * copy from disagreeing with this one about the same run.
 */

import { describe, expect, it } from "vitest";

import { formatMs, money, unavailableReason } from "@/lib/format";

describe("formatMs", () => {
  it("rounds sub-second durations to whole milliseconds", () => {
    expect(formatMs(0)).toBe("0ms");
    expect(formatMs(840)).toBe("840ms");
    expect(formatMs(12.4)).toBe("12ms");
    expect(formatMs(12.5)).toBe("13ms");
  });

  it("switches to one-decimal seconds at exactly 1000ms", () => {
    expect(formatMs(999)).toBe("999ms");
    expect(formatMs(1000)).toBe("1.0s");
    expect(formatMs(1234)).toBe("1.2s");
    expect(formatMs(95_000)).toBe("95.0s");
  });
});

describe("money", () => {
  it("says 'free' for a local run rather than printing $0.0000", () => {
    // The whole point of the local backend: the column should read as a fact
    // about the run, not as a rounded-to-zero price.
    expect(money(0)).toBe("free");
  });

  it("shows four decimals, because a Haiku call costs less than a cent", () => {
    expect(money(0.0012)).toBe("$0.0012");
    expect(money(1)).toBe("$1.0000");
  });

  it("rounds a spend too small to show down into '$0.0000', not 'free'", () => {
    // A real but tiny paid call must still read as paid. The branch is on the
    // value, not on the formatted string, which is what keeps these apart.
    expect(money(0.00001)).toBe("$0.0000");
  });
});

describe("unavailableReason", () => {
  // Only the three fields the function reads. A full `Technique` would drag the
  // API shape into a test about a branch.
  const t = (over: Partial<Parameters<typeof unavailableReason>[0]> = {}) => ({
    implemented: true,
    docs_only: false,
    needs_human: false,
    ...over,
  });

  it("says a runnable technique has no reason at all", () => {
    expect(unavailableReason(t(), "playground")).toBeNull();
    expect(unavailableReason(t(), "compare")).toBeNull();
  });

  it("does not call a docs-only technique 'not built yet'", () => {
    // The bug this function exists for. REALM is a pre-training method, so a
    // roadmap promise is a false claim, not just clumsy wording — and REALM is
    // also unimplemented, so the branch order is what makes this right.
    const realm = t({ implemented: false, docs_only: true });

    for (const context of ["playground", "compare"] as const) {
      const reason = unavailableReason(realm, context);
      expect(reason).toContain("cannot run");
      expect(reason).not.toContain("not built");
    }
  });

  it("still says 'not built yet' for a technique that merely has no pipeline", () => {
    const reason = unavailableReason(t({ implemented: false }), "playground");
    expect(reason).toContain("not built");
    expect(reason).not.toContain("cannot run");
  });

  it("blocks a human-in-the-loop technique only in the compare view", () => {
    // Interactive RAG runs fine in the playground; only an unattended side-by-side
    // has no honest result for it. Getting this backwards would either hide a
    // working technique or compare a stubbed-out human.
    const interactive = t({ needs_human: true });

    expect(unavailableReason(interactive, "playground")).toBeNull();
    expect(unavailableReason(interactive, "compare")).toContain("human");
  });
});
