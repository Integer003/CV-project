<script>
	import { Button } from "flowbite-svelte";
	import { GithubSolid, FilePdfSolid } from "flowbite-svelte-icons";

  const color = "alternative";
</script>

<nav>
	<div id="inner">
		<div class="left">
			<!-- <img src="logo.svg" alt="logo" /> -->
			<span id="title-desc">
				Interact with a dozen of <b>C</b>onditional <b>V</b>ariational <b>A</b>uto<b>e</b>ncoders
				(<b>C-VAE</b>) in your browser!
			</span>
		</div>
		<div class="right">
			<Button
				size="xs"
				href="https://github.com/Integer003/CV-project"
        target="_blank"
        {color}
        outline
				>Code
				<GithubSolid size="md" class="ml-1" />
			</Button>
			<!-- <Button
				size="xs"
				href="https://arxiv.org/abs/2409.09011"
        target="_blank"
        {color}
        outline
				>Paper
				<FilePdfSolid size="md" class="ml-1" />
			</Button> -->
		</div>
	</div>
</nav>

<style>
	nav {
		background-color: black;
	}
	#inner {
		padding: 10px;
		padding-left: 15px;
		display: flex;
		justify-content: space-between;
		height: 60px;
	}
	.left {
		display: flex;
		gap: 20px;
		align-items: center;
	}
	.right {
		/* position: fixed;
		top: 10px;
		right: 10px; */
	}
	#title-desc {
		color: ivory;
		font-size: 20px;
		font-weight: 250;
	}
	b {
		font-weight: 700;
	}
	/* #github {
		cursor: pointer;
	} */
</style>
