# Historical QUAL-002 Disposition v1.0

CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION

**DECISION-R2-02 — APPROVED BY TERRY (DISPOSITION DIRECTION ONLY) — Preserve Historical QUAL-002 and Prepare Future-Execution Supersession.**

## Historical facts

Terry 已授權 QUAL-002 並已物化 authorization。原 State 2 記錄為 AUTHORIZATION MATERIALIZED / EXECUTION HOLD。正式 P0–P9 尚未完成，沒有正式 one-shot 或已確認 canonical root 建立紀錄，現有證據支持 Run ID 未消耗。A3/A4 transcripts、舊授權、M4、原 41 項 frozen identities 與歷史 interpretation MUST 保留。Git .3 不符合目前安全底線。

## Proposed event

在已批准的 R2-02 處置方向下，正式 supersession event 的物化、生效與 production dispatch deny MUST 通過獨立驗收及另行明確授權的事件發布／治理 gate。獲准後 MUST 在新版本化 append-only 治理事件記錄 **AUTHORIZED / NOT EXECUTED / EXECUTION HOLD / RUN ID UNCONSUMED / SUPERSEDED FOR FUTURE EXECUTION**。事件 MUST 包含 event ID、Terry 批准證據定位、UTC 生效時間、原 Run ID、原 State 2 commit、原 AUTH_SHA 的引用、consumption 證據定位、前事件連結、未來 dispatch deny、R2 路線參考與事件狀態。事件本身 MUST 經驗證之人類裁決及 GitHub 變更授權才得落地。

正式 supersession event 尚未物化或接受，production dispatch deny 尚未完成驗收。原 State 2 與舊授權 bytes 保持，QUAL-002 維持 HOLD，沒有新增執行許可。

生效後，supersession 只關閉未來使用舊路線執行的資格。既有授權文件、歷史 State 2、未消耗事實及原證據 MUST 原樣保留。不得改記 scientific FAIL、取消、已執行或已消耗。正式控制器 MUST fail closed，且不得允許新舊兩 lane 同時有可啟動的有效 lease/GO。若新 lane 未進入運作批准，兩者皆 HOLD。
