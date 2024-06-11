from blade import Blade


blade = Blade(
    model_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/LDA/passage/rosie_lg_lda_1_20", #"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/filtering/rosie_1_20",
    lang = "EN",
)

blade.active_learning_loop()
