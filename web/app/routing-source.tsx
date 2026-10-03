type RoutingSummary = {
  destination: string;
  used_fallback: boolean;
};

export function routingSourceLabel(routing: RoutingSummary): string {
  if (routing.destination === "blocked") {
    return "Blocked by OpenAI Moderation";
  }
  if (routing.used_fallback) {
    return "OpenAI fallback used";
  }
  return "Jev route used";
}

export function RoutingSource({ routing }: { routing: RoutingSummary }) {
  return <small>{routingSourceLabel(routing)}</small>;
}
