import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { RoutingSource } from "./routing-source";

test("renders moderation blocks without claiming Jev ran", () => {
  const markup = renderToStaticMarkup(
    <RoutingSource routing={{ destination: "blocked", used_fallback: false }} />,
  );

  assert.equal(markup, "<small>Blocked by OpenAI Moderation</small>");
});

test("renders OpenAI fallback routing", () => {
  const markup = renderToStaticMarkup(
    <RoutingSource routing={{ destination: "billing", used_fallback: true }} />,
  );

  assert.equal(markup, "<small>OpenAI fallback used</small>");
});

test("renders direct Jev routing", () => {
  const markup = renderToStaticMarkup(
    <RoutingSource
      routing={{ destination: "tech_support", used_fallback: false }}
    />,
  );

  assert.equal(markup, "<small>Jev route used</small>");
});
