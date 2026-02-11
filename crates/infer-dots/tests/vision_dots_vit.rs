use deepseek_ocr_infer_dots::vision::dots_vit::SequenceLayout;

#[test]
fn layout_positions_follow_merge_groups() -> anyhow::Result<()> {
    let layout = SequenceLayout::from_grid(&[[1, 4, 4]], 2)?;
    assert_eq!(layout.total_tokens, 16);
    assert_eq!(layout.merge_groups, 4);
    let expected = [
        [0u32, 0u32],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ];
    assert_eq!(&layout.positions()[..8], &expected);
    Ok(())
}
