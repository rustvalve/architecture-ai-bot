# База знаний для тестирования RAG

## Логика подмены

Подмены сгенерированы через AI. Подстановки делались по такой логике:

1. Стиль «на один мир»
   Нужна была одна вымышленная вселенная, чтобы имена не выглядели случайным набором. Выбрал общий sci‑fantasy тон: короткие, «звучные» слова, без явных земных ассоциаций.
2. Категории и шаблоны
   - Персонажи — два слова (имя + фамилия/титул), похоже на человеческие имена, но не реальные: Kael Draven, Xarn Velgor, Theron Vex, Renn Korval, Sera Valdris.
   - Планеты/места — одно-два слова, «географический» вид: Solrath, Axiom Prime, Frostfall, Murkwood, Emberfall, Verdant.
     Технологии/корабли — образ по функции: Void Core (вместо Death Star), Flux Blade (вместо lightsaber), Star Drifter (корабль), Strike-wing, Shadow Dart.
   - Организации — «серьёзные» названия: Dominion of Krath, Sundered Pact, Aetherian Order, Umbral (для Sith).
   - Концепции — нейтральные термины: Synth Flux (сила/энергия), Umbral Veil (тёмная сторона), Radiant Path (светлая).
   - Расы — одно слово или два: Thornwarden, Verdantkin, Luminar, Magnate.
3. Связность
   - Одна «сила» → один термин: Synth Flux везде (и «the Synth Flux», и «Synth Flux»).
     «Светлая/тёмная» пара: Radiant Path и Umbral Veil.
   - Орден «джедаев» → Aetherian (Aetherian Order, Aetherian Knight, Aetherian Master), «ситхи» → Umbral (Umbral Lord и т.д.).
4. Регистр и производные
   - В словарь добавлены разные формы: «Jedi» и «jedi», «The Force» и «the Force», «Darth Vader» и отдельно «Vader», «Luke Skywalker» и «Skywalker», чтобы замены срабатывали в любом контексте.
   - Длинные фразы шли перед короткими (например, «Darth Vader» раньше «Darth»), чтобы не ломать составные имена.
