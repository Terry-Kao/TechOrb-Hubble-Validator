# Model B Normative Specification v1.0

CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION

MUST 為強制條款。Model B 需新 R2 規範批准。原 QUAL-002 仍依原 State 2 與 v1.1 規範解讀。

1. B 是 Accepted Technical Source Baseline。B MUST 單親接於明文核准的 policy anchor P 或另經批准的 topology，保留歷史 main 及 M4 ancestry。MUST 驗證 exact commit、tree、parent 和批准的 technical path/mode/blob diff。
2. E 是 Accepted Execution Source Commit。E MUST 為 B 直接單親子提交。預設唯一差異為指定 release manifest 的 100644 新增。擴充 freeze paths 或其他 topology 必須先改規範並重新獨立驗證。Manifest MUST 封存全部 active 文件，且不列 E、自身 digest 或未來 G。
3. G 是經獨立治理接受並與 authoritative ref 同一 exact SHA 的 Trusted Governance Head。僅 G-AUTH phase 允許宣稱已授權且 execution HOLD。G MUST 透過可信決策與遠端驗證指向 B/E/tree、授權 hash 及當次 run。信任取得依 TRUSTED_G_SOURCE.md。
4. E 到 G MUST 每層驗證單親、tree、mode 與完整 path delta。預設僅固定 R2 versioned authorization event path 可變動。不得改 source、修改歷史紀錄、加入可執行 mode、symlink、submodule，或借 revert 隱蔽中間 drift。若需新增其他治理檔，須另行版本化 allowlist 與測試。G-POLICY 放在 B 之前，並不位於 E→G-AUTH 窄 allowlist。
5. 正式工具 MUST 在專用 clean detached checkout 執行精確的 E。先以 trust envelope 取得 G→E，再核對 E→B、M4、committed/working/index identity、ignored/untracked、case/reparse 及 loaded origin。Detached HEAD、分支名或本機 SHA 只提供狀態觀測。
6. GitHub 新 G、trust record、source、raw auth、tool 或 lease 發生變更，MUST HOLD，重新建立 binding 與當次 preflight。修訂需新的版本化治理接受。歷史 G 與合成 G 均不得授予正式執行權。
7. Production bootstrap/controller 尚待驗收。文件及技術審查均保持 qualification HOLD。
