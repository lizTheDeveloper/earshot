# Live pooling — design note (not built in v1)

v1 pools by people swapping log files. A live version would let listeners share events as they happen.
Constraints carried over from v1: only event summaries ever leave a device; no audio; nothing central
that has to be trusted with everyone's positions.

## Options

1. **A Matrix room per group.** Each listener posts its event record as a message (a custom event type,
   e.g. `org.earshot.event`). Members already run Matrix clients; end-to-end encryption comes free; the
   group controls membership. A small bot (or any member's merge) clusters events and posts the
   result. Positions: each member posts their own once, encrypted to the room, and can leave any time.
   Cost: a Matrix account per listener (or one shared bot account), and a homeserver.
2. **LAN only (mDNS + UDP multicast).** Listeners on the same network find each other and share events
   with no server and no accounts. Good for one building or a campus; useless across a neighbourhood.
3. **A shared folder** (Syncthing, a git repo). Each listener appends to its own log; merge reads them
   all. Zero new code, minutes of delay.

## Recommendation

Start with (3), which v1 already supports, then (1). Matrix gives membership, encryption and history
without earshot running a server.

## Things to decide

- Who in a group may see listeners' positions, and whether positions are coarsened (e.g. to 100 m).
- Retention: how long event history is kept in the room.
- Clock sync: area estimates need clocks within ~0.1 s. Listeners should report their NTP offset with
  each event, and merge should down-weight unsynced ones.
- Abuse: a listener network that can say "a drone was over this street at 2:14 am" can be pointed at
  other sounds. Keep the detector narrow, publish the code, and keep the event record minimal.
