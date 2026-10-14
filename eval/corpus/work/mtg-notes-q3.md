# Billing DB migration - planning mtg (Q3)

Date: Aug 14, 2024. Attendees: Ines Fontaine (eng lead), Tom Wardell (DBA contractor), Kenji Aoyama, me, Lucia from finance ops (first 20 min only).

Context: orders and invoices are still on MySQL 5.7. Support ends next year. Agreed to move to Postgres 16 on RDS, cutover window Oct 26-27 (weekend, 4 hour window).

Decisions:
- logical replication for the initial sync, not dump and restore. Tom estimates 6 days of replication lag at current write volume (about 1,900 writes/sec peak).
- keep pgbouncer in front of the new cluster, same as the current proxy. Transaction pooling mode. Kenji flagged that prepared statements break under transaction pooling, so we audit the ORM config.
- money columns: DECIMAL(12,2) becomes NUMERIC(12,2). Lucia insisted on a row-by-row checksum over the last 90 days of invoices before cutover.
- no stored procedures. Three in the legacy schema, all used by the nightly report job. Port them to plain SQL views.

Open questions:
- who owns the rollback script? Tom can write it but won't be on call the 26th. Ines to find a second person.
- keep the MySQL instance read-only for 30 days after cutover? Yes, agreed, about $140/month.
- timezone handling: created_at is stored in local time in some rows from the 2019 import. Normalize to UTC before or after the move? Ines wants before. Not settled.

Action items:
1. Kenji: audit ORM for prepared statements, by Aug 30
2. Tom: replication runbook and a dry run in staging, by Sep 13
3. Me: draft the cutover checklist and send it to the group
4. Lucia: confirm the 90-day checksum query and who signs off
5. Ines: find the rollback owner

Next sync: Aug 21, same room.
