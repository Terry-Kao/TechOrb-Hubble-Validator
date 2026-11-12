# Governance and Authorization Binding v1.0

CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION

G-POLICY MUST 只採納政策及 acceptance criteria。禁止配置正式 Run ID、寫入 AUTH_SHA、物化 auth 或允許 launch。G-POLICY 決策可留在新的 append-only scope，與歷史 State 2 分開。

G-AUTH MUST 在核准新身分、完成必要 A0、接受正式 B/E 後，才由獨立 Terry 授權並物化外部 raw JSON。Authorization MUST 綁 R2 profile、scope、one-shot/no-retry、run、E、B、protocol/preregistration、manifest/environment SHA、runner/provenance blobs、固定 governance path、issuer、UTC。原 v1.1 規範禁止 execution_head，新 R2 欄位須獨立接受。

G-AUTH event MUST 綁當次 Run ID、B/E commit/tree、frozen identities、raw AUTH_SHA、版本化 path、Terry 決策、EXECUTION HOLD 與下一 gate。它 MAY 引用已核准資料的 exact immutable object identities。G 產生後才可簽獨立接受 receipt。正式 G-AUTH 未能證明時一律 HOLD。

P8 MUST 比較當次 transcript AUTH_SHA 欄位、G-AUTH AUTH_SHA 欄位、外部 auth 原始 bytes 的 SHA-256。禁止重排 JSON、使用治理檔自身 hash、填 placeholder hash 或套用舊 QUAL-002 AUTH_SHA。G 的 SHA 不得寫入它自己、E manifest 或需先物化的 auth。
