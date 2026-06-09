import os
import sys

def main():
	import argparse
	parser = argparse.ArgumentParser(description='Image captioning CLI')
	sub = parser.add_subparsers(dest='cmd')

	t = sub.add_parser('train')
	t.add_argument('--data-dir', required=True)
	t.add_argument('--captions', default='captions.txt')
	t.add_argument('--epochs', type=int, default=5)
	t.add_argument('--save-path', default='models/caption.pth')

	i = sub.add_parser('infer')
	i.add_argument('--image', required=True)
	i.add_argument('--model', required=True)

	u = sub.add_parser('ui')
	u.add_argument('--port', type=int, default=5000)

	args = parser.parse_args()
	if args.cmd == 'train':
		from train import train
		captions_file = args.captions
		if not os.path.isabs(captions_file) and not os.path.exists(captions_file):
			captions_file = os.path.join(args.data_dir, args.captions)
		if not os.path.exists(captions_file):
			print(f"Error: captions file not found: {captions_file}")
			sys.exit(1)
		train(args.data_dir, captions_file, epochs=args.epochs, save_path=args.save_path)
	elif args.cmd == 'infer':
		from inference import caption_image
		print(caption_image(args.image, args.model))
	elif args.cmd == 'ui':
		from app import app
		app.run(host='0.0.0.0', port=args.port, debug=True)
	else:
		parser.print_help()


if __name__ == '__main__':
	main()

