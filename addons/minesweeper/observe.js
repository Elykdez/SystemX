function () {
    const board = document.querySelector('table#board[aria-label="Minesweeper board"]');
    if (!board) throw new Error('Visible Minesweeper board was not found');
    const cells = Array.from(board.querySelectorAll('tbody > tr > td > button.cell'), button => {
        const rect = button.getBoundingClientRect();
        const top = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
        return {
            label: button.getAttribute('aria-label'),
            box: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
            hit: top === button || (top !== null && button.contains(top))
        };
    });
    const difficulty = document.querySelector('select#difficulty');
    return {
        cells,
        statusText: document.querySelector('#announce[role="status"]')?.textContent.trim() ?? '',
        difficultyText: difficulty?.selectedOptions[0]?.textContent.trim() ?? '',
        faceText: document.querySelector('button#new-game')?.textContent.trim() ?? '',
        viewport: {width: window.innerWidth, height: window.innerHeight}
    };
}
