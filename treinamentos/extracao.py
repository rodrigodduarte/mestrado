def _estimate_model_size_mb(model: torch.nn.Module):
    """
    Estima o tamanho (em MB) ocupado pelos pesos do modelo em float atual.
    Isso substitui 'checkpoint_size_mb', já que não estamos salvando .ckpt.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def benchmark_single_fold(h, fold_idx, base_save_dir, seed=42):
    """
    Executa benchmark de UM modelo (h['TMODEL']) em UM fold:
    - treina 1 época (sem salvar checkpoint)
    - mede custo de treino e custo de inferência
    - mede métricas de validação e teste
    - mede memória de GPU e throughput
    - salva SOMENTE estatísticas (.json e .txt), nenhum modelo

    Retorna um dict 'report' com tudo.
    """

    # reproducibilidade
    set_seeds(seed)

    dataset_name = h["NAME_DATASET"]
    tmodel_name = h["TMODEL"]

    # diretórios de saída
    run_name = f"{dataset_name}_{tmodel_name}"
    run_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # salva hiperparâmetros usados (snapshot do config)
    with open(os.path.join(fold_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    # monta datamodule pro fold
    k_folds = int(h.get("K_FOLDS", 5))
    dm, dm_type = build_datamodule(h, n_splits=k_folds, fold_idx=fold_idx)

    dm.setup(stage="fit")

    # instancia modelo Lightning
    model = build_model(h, dm_type, dm=dm)

    # Trainer SEM checkpoint callback
    trainer = pl.Trainer(
        log_every_n_steps=10,
        accelerator=h["ACCELERATOR"],
        devices=h["DEVICES"],
        precision=h["PRECISION"],
        max_epochs=1,  # só 1 época
        callbacks=[TQDMProgressBar(leave=True)],
    )

    # ---------- TREINO ----------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.fit(model, dm)
    train_time_sec = time.perf_counter() - t0

    max_gpu_mb_train = None
    if torch.cuda.is_available():
        max_gpu_mb_train = torch.cuda.max_memory_allocated() / (1024**2)

    # "época treinada": no nosso benchmark é sempre só a epoch 0
    epoch_trained = 0

    # ---------- VALIDAÇÃO ----------
    # usamos o próprio modelo já treinado em memória
    val_metrics_list = trainer.validate(model, datamodule=dm, verbose=False)
    val_metrics = val_metrics_list[0] if len(val_metrics_list) > 0 else {}

    # ---------- TESTE ----------
    try:
        dm.setup(stage="test")
    except Exception:
        pass

    test_dl = dm.test_dataloader()
    n_test = len_dataloader_safe(test_dl)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t1 = time.perf_counter()
    test_metrics_list = trainer.test(model, datamodule=dm, verbose=False)
    test_time_sec = time.perf_counter() - t1

    test_metrics = test_metrics_list[0] if len(test_metrics_list) > 0 else {}

    max_gpu_mb_test = None
    if torch.cuda.is_available():
        max_gpu_mb_test = torch.cuda.max_memory_allocated() / (1024**2)

    # métricas derivadas do tempo de teste
    if n_test > 0 and test_time_sec > 0:
        inf_ms_per_sample = (test_time_sec * 1000.0) / n_test
        throughput_samples_per_sec = n_test / test_time_sec
    else:
        inf_ms_per_sample = float("nan")
        throughput_samples_per_sec = float("nan")

    # ---------- TAMANHO DO MODELO (APROX) ----------
    # estimativa do tamanho dos pesos em MB
    model_size_mb = _estimate_model_size_mb(model)

    # número de parâmetros treináveis
    num_params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # ---------- REPORT FINAL ----------
    report = {
        "dataset": dataset_name,
        "tmodel": tmodel_name,
        "fold_idx": fold_idx,

        "epoch_trained": epoch_trained,  # substitui best_epoch
        "train_time_sec": train_time_sec,
        "test_time_sec": test_time_sec,

        "test_inf_ms_per_sample": inf_ms_per_sample,
        "throughput_samples_per_sec": throughput_samples_per_sec,

        "max_gpu_mem_mb_train": max_gpu_mb_train,
        "max_gpu_mem_mb_test": max_gpu_mb_test,

        # antes era checkpoint_size_mb; agora é tamanho estimado dos pesos na RAM
        "model_size_mb": model_size_mb,
        "num_params_trainable": num_params_trainable,

        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # ---------- SALVAR ESTATÍSTICAS (sem pesos!) ----------
    # json bruto
    with open(os.path.join(fold_dir, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # txt humano
    with open(os.path.join(fold_dir, "benchmark_result.txt"), "w") as f:
        f.write("===== BENCHMARK REPORT (fold individual) =====\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] {dataset_name}/{tmodel_name}/fold_{fold_idx} concluído (sem salvar modelo).")
    return report
