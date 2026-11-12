# Trusted G Source Contract v1.0

CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION

1. 正式控制器 MUST 從獨立核准的 deployment policy 載入固定 GitHub host、repository 名稱及由批准時另行核對的 immutable repository ID、refs/heads/main、批准簽署者信任根、受控 Git/TLS 及本機隔離路徑。空白、redirect 到不同 repository、owner/name 與 ID 不合均 HOLD。
2. 使用受控唯讀 GitHub API 驗 repository ID、branch protection/ruleset 生效與 exact ref SHA。獨立以 pinned Git/TLS 取得同一遠端 ref 到隔離暫存 ref。兩讀 SHA MUST 相等，且 fetched object/tree MUST 驗證。依賴本機 main、origin/main 快取、任意 CLI、JSON/env 提供 pin，或在 repo 檔內自稱批准，一律拒絕。
3. MUST 另驗獨立且可稽核的人類接受證據。建議用預先註冊的 Terry 治理驗章公鑰，驗外部 detached approval receipt 的 raw 簽章。Receipt 綁 repository immutable ID、ref、phase、G、B/E/tree、decision IDs、auth digest（僅 G-AUTH）、前 accepted checkpoint、單調序號、有效期與決策定位。正式 trust registry 的批准、輪替、撤回與保管 MUST 獨立審核。GitHub 上的核准紀錄與 G 內指向的 event MUST 互相核對。Synthetic issuer、測試信任根及 review profile 永遠不能成為正式 receipt。
4. 只有遠端 ref、fetched Git object、receipt 與核准決策一致時才有 G pin。拒絕非快轉、序號回退、撤回、force-push、unknown ancestry、split view、API/Git disagreement、網路或簽章錯誤。不得以快取值繼續。控制器需將最新接受 checkpoint 保存於不可被普通 checkout 改寫的 durable store。
5. 在 fresh P0、lease 內與立即 GO 前重新觀測 API、Git 與撤回狀態。任何 drift 即作廢舊 seal/GO/ticket，重新完整 preflight。遠端檢查與本機 launch 無分散式原子性，正式部署 MUST 有排他治理發布與啟動窗口，並定義 immutable dispatch permit、撤回 cutoff 及證據記錄。無可驗證的窗口控制即 HOLD。
6. 本包 validator 用兩個實際隔離 Git refs 和獨立固定的合成 receipt digest 模擬此契約；正式網路信任、簽章驗收及窗口控制仍待實作。
